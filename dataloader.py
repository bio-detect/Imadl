from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import glob

class myDataset(Dataset):

    def __init__(self,file_pathname):
        self.file_pathname = file_pathname
        self.txt_path = []
        for filename in os.listdir(file_pathname):
            path = os.path.join(file_pathname, filename)
            self.txt_path.append(path)

    def __getitem__(self, index):
        raw = []
        label = []
        label_p = []
        path = self.txt_path[index]
        txt_name = os.path.basename(path)
        with open(path) as f:
            for lines in f:
                data = lines.strip('\n').split()
                raw_data = [float(i.strip("'")) for i in data[0:6]]
                raw.append(raw_data)
               
                label_p.append(data[6].strip('.'))
               
        if str(1) in label_p:
            label.append(1)
        else:
            label.append(0)

        return torch.tensor(raw,dtype=torch.float), torch.tensor(label,dtype=torch.long)


    def __len__(self):

        return len(self.txt_path)



class myDataset_test(Dataset):

    def __init__(self,file_pathname,data_file):
        self.file_pathname = file_pathname
        self.data_file = data_file
        self.txt_path = []
        with open(self.data_file) as p:
            for lines in p:
                line = lines.strip('\n')
                path = os.path.join(self.file_pathname, line)
                self.txt_path.append(path)
        '''for filename in os.listdir(self.file_pathname):
            path = os.path.join(self.file_pathname, filename)
            if filename in self.data_file:
                self.txt_path.append(path)'''

    def __getitem__(self, index):
        raw = []
        label = []
        label_p = []
        path = self.txt_path[index]
        txt_name = os.path.basename(path)
        file_name = txt_name.replace('_raw', '', 1)
        with open(path) as f:
            for lines in f:
                data = lines.strip('\n').split()
                raw_data = [float(i.strip("'")) for i in data[0:6]]
                raw.append(raw_data)
               
                label_p.append(data[6].strip('.'))
        if str(1) in label_p:
            label.append(1)
        else:
            label.append(0)

        return torch.tensor(raw,dtype=torch.float), torch.tensor(label,dtype=torch.long)


    def __len__(self):

        return len(self.txt_path)

 

