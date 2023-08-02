from torch.utils.data import Dataset
import joblib

class ArgoData(Dataset):
    def __init__(self, 
                 files
    ):
        super().__init__()
        self.files = files
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        data = joblib.load(file_path)
        return data
