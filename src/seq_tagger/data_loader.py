from torch.utils.data import Dataset
import torch


class load_dataset(Dataset):
    def __init__(self,samples,labels,device):
        self.len = len(labels)
        try:
            self.samples = torch.tensor(samples,device=device,dtype=torch.long)
        except:
            self.samples = [torch.tensor(s,device=device,dtype=torch.long).view(-1,1) for s in samples]
        self.labels = torch.tensor(labels,device=device,dtype=torch.long)
        pass

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass