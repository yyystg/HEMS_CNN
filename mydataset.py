from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import csv
import torch

class CustomDataset(Dataset):
    def __init__(self,args,csv_path,transform=None):
        # Transforms
        self.transform = transform
        # Read the csv file
        self.data1 = np.asarray(pd.read_csv(csv_path,header=None))
        self.info1 = self.data1[:, :9]
        self.data1 = self.data1[:, 9:]
        self.data1 = self.data1.reshape(-1, 31, 15)
        self.input = self.data1
        self.target = self.info1[:,-5:-2]


        self.data_len = len(self.info1)

    def __getitem__(self, index):
        # item_size = len(self.data1.columns[:])

        x = self.input[index]
        y = self.target[index]
        info = self.info1[index]
        if self.transform is not None:
            x = self.transform(x)

        return x,y,info

    def __len__(self):
        return self.data_len

    def normalization(self):
        target = np.asarray(self.target,dtype=float)
        target=target-self.out_min
        target=target/(self.out_max-self.out_min)

        return target