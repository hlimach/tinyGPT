import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ single self-attention head"""

    def __init__(self, n_embd, block_size, head_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # generate key, query, and value for this head
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        # compute attention scores by matrix multiplication of q and k and normalizing
        k = k.transpose(-2, -1) # (B, C, T)
        w = q @ k # (B, T, C) @ (B, C, T) -> (B, T, T)

        # normalize for scaled attention
        w = w * C**-0.5 

        # change all zeroes in tril to -inf
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # compute softmax on each row and dropout
        w = F.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w)

        # weighted aggregation of values
        out = w @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeads(nn.Module):
    """ multiple self-attention heads in parallel for communication"""

    def __init__(self, n_embd, block_size, head_size, dropout, n_heads):
        super().__init__()

        self.heads = nn.ModuleList([Head(n_embd, block_size, head_size, dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(x))
        return out


class FeedForward(nn.Module):
    """ simple feedforward block for computation"""

    def __init__(self, n_embd, dropout):
        super().__init__()

        self.feed = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.feed(x)


class Block(nn.Module):
    """ transformer block for communication then computation"""

    def __init__(self, n_embd, block_size, dropout, n_heads):
        super().__init__()
        head_size = n_embd // n_heads

        self.self_attn_heads = MultiHeads(n_embd, block_size, head_size, dropout, n_heads)
        self.feedfwd = FeedForward(n_embd, dropout)

        # for pre-norm formulation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.self_attn_heads(self.ln1(x)) # (B, T, C)
        out = x + self.feedfwd(self.ln2(x)) # (B, T, C)
        return out


class GPTLanguageModel(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.vocab_size = params['vocab_size']
        self.block_size = params['block_size']
        self.n_layers = params['n_layers']
        self.dropout = params['dropout']
        self.n_heads = params['n_heads']
        self.n_embd = params['n_embd']
        self.device = params['device']

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.pos_embedding_table = nn.Embedding(self.block_size, self.n_embd)

        self.transformer = nn.Sequential(
            *[Block(self.n_embd, self.block_size, self.dropout, self.n_heads) for _ in range(self.n_layers)]
        )

        self.ln = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        
        # logits generated with size (batch, time, channels)
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=self.device)) # (T, C)

        x = tok_emb + pos_emb # (B, T, C)
        x = self.ln(self.transformer(x)) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if target is None:
            loss = None
        else:
            # reshape for cross entropy loss calculation
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)

            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        # idx is an int tensor of size (batch B, time T)

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            # get prediction on current idx
            logits, loss = self(idx_cond) # (B, T, C)

            # get only the last timestep
            logits = logits[:, -1, :] # (B, C)

            # use last time step to generate probabilities
            prob = F.softmax(logits, dim=-1) # (B, C)

            # generate next value, sampled from probability distribution
            next_idx = torch.multinomial(prob, num_samples=1) # (B, 1)

            # stack it to running idx sequence
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)

        return idx


