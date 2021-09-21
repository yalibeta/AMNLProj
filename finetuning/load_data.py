import pickle

from torch.utils.data import Dataset
from transformers import T5Tokenizer


class MyDataset(Dataset):
    def __init__(self, path, size):
        self.arr = pickle.load(open(path, 'rb'))[:size]

    def __getitem__(self, item):
        return self.arr[item]

    def __len__(self):
        return len(self.arr)
