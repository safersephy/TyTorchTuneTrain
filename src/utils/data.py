import torch
from torch.nn.utils.rnn import pad_sequence

def add_batch_padding(batch):
    X, y = zip(*batch)  
    X_ = pad_sequence(X, batch_first=True)      
    return X_, torch.tensor(y)