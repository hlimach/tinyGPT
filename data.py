import torch
from torch.utils.data import Dataset

class CharData():
    def __init__(self, text, params):
        self.text = text

        # build vocab from given text data
        self.chars = self.build_vocab()
        self.vocab_size = self.get_vocab_size()

        # create encoder decoder mappings for char-level data
        self.stoi = {c:i for i, c in enumerate(self.chars)}
        self.itos = {i:c for i, c in enumerate(self.chars)}

        # create tensors for train and test data
        self.data = torch.tensor(self.encode(self.text))
        self.split = int(params['split']*len(self.data))
        
        self.train = self.data[:self.split]
        self.val = self.data[self.split:]

        # to create batches of data
        self.batch_size = params['batch_size'] # for parallel processing
        self.block_size = params['block_size'] # max context length for pred
    

    def build_vocab(self):
        return list(set(self.text))


    def get_vocab_size(self):
        return len(self.chars)


    def encode(self, string):
        return [self.stoi[c] for c in string]


    def decode(self, ints):
        return ''.join([self.itos[i] for i in ints])


    def get_batch(self, split):
        data = self.train if split == 'train' else self.val
        idx = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in range(idx)])
        y = torch.stack([data[i:i+self.block_size] for i in range(1,idx+1)])
        return x, y