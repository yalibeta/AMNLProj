import pickle

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, size=1):
        self.arr = pickle.load(open(path, 'rb'))
        n = int(len(self.arr)*size)
        self.arr = self.arr[:n]

    def __getitem__(self, item):
        return self.arr[item]

    def __len__(self):
        return len(self.arr)
