import pandas as pd
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        if isinstance(annotations, str):
            self.img_labels = pd.read_csv(annotations)
        else:
            self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        label = int(label)  # Ensure label is an integer
        return image, label

def load_data(csv_file, img_dir, batch_size=32):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = pd.read_csv(csv_file)
    train_data = full_dataset[full_dataset['category'] == 'train']
    val_data = full_dataset[full_dataset['category'] == 'val']

    train_dataset = CustomImageDataset(annotations=train_data, img_dir=img_dir, transform=data_transforms['train'])
    val_dataset = CustomImageDataset(annotations=val_data, img_dir=img_dir, transform=data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = train_data['HER2score'].unique()

    return dataloaders, dataset_sizes, class_names
