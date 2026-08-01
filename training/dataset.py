# Additional data processing, transformations, loaders

import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class PadToSquare:
    def __init__(self, fill=255): 
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

def get_transforms(image_size=300):
    train_transforms = transforms.Compose([
        PadToSquare(fill=255),                  
        transforms.Resize((image_size, image_size)),          
        transforms.RandomHorizontalFlip(),      
        transforms.ToTensor(),                  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        PadToSquare(fill=255),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, eval_transforms

class AirlinerDataset(Dataset):
    def __init__(self, dataframe, image_dir, airline_to_idx, variant_to_idx, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.airline_to_idx = airline_to_idx
        self.variant_to_idx = variant_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = str(row['image_filename'])
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"Skipping {img_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            image = self.transform(image)
            
        airline_str = str(row['airline'])
        variant_str = str(row['aircraft_variant'])
        
        airline_label = self.airline_to_idx.get(airline_str, self.airline_to_idx.get('OTHERS', 0))
        variant_label = self.variant_to_idx.get(variant_str, self.variant_to_idx.get('OTHERS', 0))
        
        return image, torch.tensor(variant_label, dtype=torch.long), torch.tensor(airline_label, dtype=torch.long)

def load_split_dataframes(train_csv_path, val_csv_path, test_csv_path):
    print("Loading generated splits from disk...")
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    train_df.fillna('OTHERS', inplace=True)
    val_df.fillna('OTHERS', inplace=True)
    test_df.fillna('OTHERS', inplace=True)
    
    return train_df, val_df, test_df

def build_mapping_from_csv(csv_path, column_name):
    df = pd.read_csv(csv_path)
    class_list = df[column_name].dropna().unique().tolist()
    if 'OTHERS' not in class_list:
        class_list.append('OTHERS')
    return {name: idx for idx, name in enumerate(class_list)}