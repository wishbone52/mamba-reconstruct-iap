from torch.utils.data import Dataset

class ApDataset(Dataset):
    def __init__(self, extras, intras,apds):
        """
        intras: shape (num_data, 8000)
        extras: shape (num_data, 8000)
        apds: shape (num_data, 10)
        """
        self.extras = extras
        self.intras = intras
        self.apds = apds
        
    def __len__(self):
        return len(self.extras)
    
    def __getitem__(self, idx):
        return self.extras[idx],self.intras[idx],self.apds[idx]