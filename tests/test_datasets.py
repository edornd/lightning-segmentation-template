import albumentations as alb
from albumentations.pytorch import ToTensorV2
from os.path import join
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.data.isprs.config import ISPRSChannels
from src.data.isprs import ISPRSDataset, PotsdamDataset, VaihingenDataset


def test_dataset_vaihingen(root_path: str):
    dataset = ISPRSDataset(path=join(root_path, "vaihingen", "train"),
                           include_dsm=True)
    image, mask = dataset.__getitem__(0)
    print(image.shape, mask.shape)
    print(image.dtype, mask.dtype)
    print(image.min(), image.max())
    print(mask.min(), mask.max())


def test_dataset_potsdam(root_path: str):
    dataset = ISPRSDataset(path=join(root_path, "potsdam", "train"),
                           channels=ISPRSChannels.RGBIR)
    image, mask = dataset.__getitem__(0)
    print(image.shape, mask.shape)
    print(image.dtype, mask.dtype)
    print(image.min(), image.max())
    print(mask.min(), mask.max())


def test_dataset_potsdam_dsm(root_path: str):
    dataset = ISPRSDataset(path=join(root_path, "potsdam", "train"),
                           channels=ISPRSChannels.RGBIR,
                           include_dsm=True)
    image, mask = dataset.__getitem__(0)
    print(image.shape, mask.shape)
    print(image.dtype, mask.dtype)
    print(image.min(), image.max())


def test_vaihingen_mean_var(root_path: str):
    preproc = alb.Compose([
        alb.Normalize(mean=VaihingenDataset.MEAN, std=VaihingenDataset.STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])
    dataset = VaihingenDataset(path=join(root_path, "vaihingen", "train"),
                               include_dsm=True,
                               transform=preproc)
    image, _ = dataset.__getitem__(0)
    print("img: ", image.view(image.size(0), -1).mean(dim=-1), image.view(image.size(0), -1).std(dim=-1))


def test_vaihingen_normalize_full(root_path: str):
    aug = alb.Compose([
        alb.Normalize(mean=VaihingenDataset.MEAN, std=VaihingenDataset.STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])
    dataset = VaihingenDataset(path=join(root_path, "vaihingen", "train"),
                               include_dsm=True,
                               transform=aug)
    loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    channels = torch.tensor([0., 0., 0., 0.], dtype=torch.float32)
    for image, mask in tqdm(loader):
        valid = image.squeeze(0)[:, mask.squeeze(0) != 255]
        channels += valid.mean(dim=-1)
    print(channels / len(dataset))


def test_potsdam_normalize_full(root_path: str):
    aug = alb.Compose([
        alb.Normalize(mean=PotsdamDataset.MEAN, std=PotsdamDataset.STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])
    dataset = PotsdamDataset(path=join(root_path, "potsdam", "train"),
                             include_dsm=True,
                             transform=aug)
    loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    channels = torch.tensor([0., 0., 0., 0., 0.], dtype=torch.float32)
    for image, mask in tqdm(loader):
        valid = image.squeeze(0)[:, mask.squeeze(0) != 255]
        channels += valid.mean(dim=-1)
    print(channels / len(dataset))


def test_potsdam_mean_var(root_path: str):
    preproc = alb.Compose([
        alb.Normalize(mean=PotsdamDataset.MEAN, std=PotsdamDataset.STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])
    dataset = PotsdamDataset(path=join(root_path, "potsdam", "train"),
                             include_dsm=True,
                             transform=preproc)
    image, _ = dataset.__getitem__(0)
    print("img: ", image.view(image.size(0), -1).mean(dim=-1), image.view(image.size(0), -1).std(dim=-1))


def mean_std_vaihingen(root_path: str):
    preprocess = alb.Compose([ToTensorV2()])
    dataset = VaihingenDataset(path=join(root_path, "vaihingen", "train"),
                               include_dsm=True,
                               transform=preprocess)
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=8,
                        shuffle=False)

    mean = torch.zeros(4, dtype=torch.float32)
    std = torch.zeros(4, dtype=torch.float32)
    pixels = 512 * 512

    batches = 0
    for images, mask in loader:
        # need channels last again, where mask is valid
        valid_pixels = images.permute(0, 2, 3, 1)[mask < 255]
        mean += valid_pixels.mean(dim=0)
        batches += 1
    mean = mean / batches

    var = 0.0
    for images, _ in loader:
        valid_pixels = images.permute(0, 2, 3, 1)[mask < 255]
        var += ((valid_pixels - mean.unsqueeze(0))**2).sum(0)
    std = torch.sqrt(var / (len(dataset) * pixels))
    print(mean)
    print(std)


def mean_std_potsdam(root_path: str):
    preprocess = alb.Compose([ToTensorV2()])
    dataset = PotsdamDataset(path=join(root_path, "potsdam", "train"),
                             include_dsm=True,
                             transform=preprocess)
    loader = DataLoader(dataset,
                        batch_size=16,
                        num_workers=8,
                        shuffle=False)
    mean = torch.zeros(5, dtype=torch.float32)
    std = torch.zeros(5, dtype=torch.float32)
    pixels = 512 * 512

    batches = 0
    for images, mask in loader:
        # need channels last again, where mask is valid
        valid_pixels = images.permute(0, 2, 3, 1)[mask < 255]
        mean += valid_pixels.mean(dim=0)
        batches += 1
    mean = mean / batches

    var = 0.0
    for images, mask in loader:
        valid_pixels = images.permute(0, 2, 3, 1)[mask < 255]
        var += ((valid_pixels - mean.unsqueeze(0))**2).sum(0)
    std = torch.sqrt(var / (len(dataset) * pixels))
    print(mean)
    print(std)
