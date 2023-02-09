import torch

class CharData():

    def __init__(self, text, params):
        self.text = text

        # build vocab from given text data
        self.chars = self.build_vocab()

        # create encoder decoder mappings for char-level data
        self.stoi = {c:i for i, c in enumerate(self.chars)}
        self.itos = {i:c for i, c in enumerate(self.chars)}

        # create tensors for train and test data
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.split = int(params['split_size']*len(self.data))
        
        self.train = self.data[:self.split]
        self.val = self.data[self.split:]

        # to create batches of data
        self.batch_size = params['batch_size'] # for parallel processing
        self.block_size = params['block_size'] # max context length for pred
        self.device = params['device']

        print('data loader successfully initiated.')
    

    def build_vocab(self):
        return sorted(list(set(self.text)))


    def get_vocab_size(self):
        return len(self.chars)


    def encode(self, string):
        return [self.stoi[c] for c in string]


    def decode(self, ints):
        return ''.join([self.itos[i] for i in ints])


    def get_batch(self, split):
        data = self.train if split == 'train' else self.val
        idx = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in idx])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in idx])
        return x.to(self.device), y.to(self.device)