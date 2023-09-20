import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
class testDataset(Dataset):
    def __init__(self,a):
        self.a = a
        with os.popen(self.a) as p:
            self.file = []
            lines = p.readlines()
            for i in range(0, len(lines), 2001):
                chunk = lines[i: i + 2001]
                if chunk[0] != '\n' and len(chunk) == 2001 and chunk[-1] != '\n':
                    self.file.append(chunk)
        '''while True:
                chunk = []
                for _ in range(8):
                    line = p.readline()
                    if len(line) == 0:
                        break
                    chunk.append(line)
                if len(chunk) == 0:
                        break
                self.file.append(chunk)'''
    def __getitem__(self, index):
        raw = []
        path = self.file[index]
        for i in range(len(path)):
            lines = path[i]
            if '>' in path[i]:
                p_name = lines.split('\t')[0].strip('>')
                pos_start = lines.split('\t')[1]
                pos_end = lines.split('\t')[2].strip('\n')
                pos_name = p_name + ':' + pos_start + '-' + pos_end
            else:
                data = lines.strip('\n').split('\t')
                raw_data = [int(j) for j in data[2:]]
                raw.append(raw_data)
        return torch.tensor(raw, dtype=torch.float), pos_name
    def __len__(self):
        return len(self.file)

 
   
