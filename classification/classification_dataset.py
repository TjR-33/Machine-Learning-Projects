from torch.utils.data import Dataset
from PIL import Image   

class PlantDataset(Dataset):
    def __init__(self, root_dir,image_list, binary_pred_list, transform=None):
        self.root_dir = root_dir
        self.image_list = image_list
        self.binary_pred_list = binary_pred_list
        self.transform = transform
        self.class_names = ['healthy', 'unhealthy']
        self.classes = [self.class_names[binary_pred] for binary_pred in self.binary_pred_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.root_dir+'/'+self.image_list[idx])
        binary_pred = self.binary_pred_list[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, binary_pred