from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_list, mask_list, transform=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        mask = Image.open(self.mask_list[idx]).convert("L")  # L 表示灰度图

        image = image.resize((768, 768))
        mask = mask.resize((768, 768))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 将掩码转换为二值图（根据你的任务调整）
        mask = (mask > 0.5).float()

        return image, mask


def load_ISIC2018(path, train_transform=None, test_transform=None):
    image_path = os.path.join(path, "ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input")
    image_list = os.listdir(image_path)
    mask_path = os.path.join(path, "ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth")
    mask_list = os.listdir(mask_path)
    for idx in range(len(image_list)):
        image_list[idx] = os.path.join(image_path, image_list[idx])
        mask_list[idx] = os.path.join(mask_path, mask_list[idx])
    X_train, X_test, y_train, y_test = train_test_split(image_list, mask_list, test_size=0.2)
    train_set = ImageDataset(X_train, y_train, train_transform)
    test_set = ImageDataset(X_test, y_test, test_transform)
    return train_set, test_set


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set, test_set = load_ISIC2018("/data/公共数据集/ISIC2018", train_transform, test_transform)
    train_loader = DataLoader(train_set)
    test_loader = DataLoader(test_set)
    for x, y in train_loader:
        print()