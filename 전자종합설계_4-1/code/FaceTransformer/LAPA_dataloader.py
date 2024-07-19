import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Optional
from pytorch_lightning import LightningDataModule

class LAPADataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        print("===============================")
        print(f"Images directory: {images_dir}")
        print(f"Labels directory: {labels_dir}")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"The directory {images_dir} does not exist.")
        
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in directory {images_dir}.")

        self.resize = transforms.Resize((256, 256))  # 이미지와 라벨을 리사이즈하는 트랜스폼 추가

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        label_name = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.png'))

        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("L")

        image = self.resize(image)  # 이미지 리사이즈
        label = self.resize(label)  # 라벨 리사이즈

        if self.transform:
            image = self.transform(image)
            label = transforms.ToTensor()(label)  # 라벨 이미지를 ToTensor로 변환

        # 라벨 값이 0에서 10 사이인지 확인 및 변환
        label = (label * 255).long()
        label[label < 0] = 0  # -1 값을 0으로 변환
        label = torch.clamp(label, min=0, max=10)  # 라벨 값을 0에서 10 사이로 클램핑

        # 디버깅 출력을 추가하여 변환된 라벨 값을 확인
        #print(f"Label transformed min: {label.min().item()}, max: {label.max().item()}")

        return image, label


class LAPASegmentationDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, image_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def setup(self, stage: Optional[str] = None):
        train_images_dir = os.path.join(self.data_dir, 'train', 'images')
        train_labels_dir = os.path.join(self.data_dir, 'train', 'labels')
        val_images_dir = os.path.join(self.data_dir, 'val', 'images')
        val_labels_dir = os.path.join(self.data_dir, 'val', 'labels')
        test_images_dir = os.path.join(self.data_dir, 'test', 'images')
        test_labels_dir = os.path.join(self.data_dir, 'test', 'labels')

        print(f"Train images directory: {train_images_dir}")
        print(f"Train labels directory: {train_labels_dir}")
        print(f"Val images directory: {val_images_dir}")
        print(f"Val labels directory: {val_labels_dir}")
        print(f"Test images directory: {test_images_dir}")
        print(f"Test labels directory: {test_labels_dir}")

        if stage == 'fit' or stage is None:
            self.train_set = LAPADataset(images_dir=train_images_dir, labels_dir=train_labels_dir, transform=self.transform)
            self.val_set = LAPADataset(images_dir=val_images_dir, labels_dir=val_labels_dir, transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_set = LAPADataset(images_dir=test_images_dir, labels_dir=test_labels_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4)


if __name__ == '__main__':
    data_dir = r'C:\Users\jaehy\FTproject\LAPA\LaPa'  # 경로 수정
    data_module = LAPASegmentationDataModule(data_dir, batch_size=32, image_size=256)
    data_module.setup('fit')

    for batch in data_module.train_dataloader():
        images, labels = batch
        print(images.shape, labels.shape)
        break
