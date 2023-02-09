import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # token table of size (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, target=None):
        # idx and targets both int tensors of size (batch B, time T)
        
        # logits generated with size (batch, time, channels)
        logits = self.token_embedding_table(idx)

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

            # get prediction on current idx
            logits, loss = self(idx) # (B, T, C)

            # get only the last timestep
            logits = logits[:, -1, :] # (B, C)

            # use last time step to generate probabilities
            prob = F.softmax(logits, dim=-1) # (B, C)

            # generate next value, sampled from probability distribution
            next_idx = torch.multinomial(prob, num_samples=1) # (B, 1)

            # stack it to running idx sequence
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)

        return idx


