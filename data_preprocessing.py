import os 
from pathlib import Path
from PIL import Image 
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self,root_dir: Path, batch_size: int = 4, shuffle: bool = True):
        self.root_dir = root_dir
        self.list_dir = os.listdir(self.root_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
        ])
  
    def get_length(self):
        for index,name in enumerate(self.list_dir):
            dir_data = os.path.join(self.root_dir,name)
            subdir_data = os.listdir(dir_data)
            for index,subname in enumerate(subdir_data):
                subdir_data = os.path.join(dir_data,subname)
                length = len(os.listdir(subdir_data))
                print(f"The folder: {name} and the class {subname} has {length} images")
    
    def get_loader(self):
        train_dataset = datasets.ImageFolder(self.root_dir + 'train-cat-rabbit', transform=self.transform)
        test_dataset = datasets.ImageFolder(self.root_dir + 'test-images', transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = self.shuffle)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = self.shuffle)    
        return train_loader, test_loader
    

    
    
        
        
# dataset = DataSet(root_dir = '/media/oscar/Seagate/machine_learning_engineer/CNNs/data/')
# train_loader, test_loader = dataset.get_loader()
# print(len(train_loader))
# # for idx, dataset in enumerate(train_loader):
# #     X = dataset[0]
# #     y = dataset[1]
# #     print(idx,X,y)
# #     break