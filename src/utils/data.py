import torch
from torch.nn.utils.rnn import pad_sequence


def add_batch_padding(batch):
    X, y = zip(*batch)  # noqa: N806
    X_ = pad_sequence(X, batch_first=True)  # noqa: N806
    return X_, torch.tensor(y)  # noqa: N806
