import torch
from torch.utils.data import Dataset, DataLoader
import medmnist
from medmnist import INFO
import torchvision.transforms as transforms

class NoisyTissueMNIST(Dataset):
    """
    Wrapper around TissueMNIST to provide (noisy_image, clean_image) pairs.
    """
    def __init__(self, split, config, image_processor):
        self.config = config
        self.image_processor = image_processor
        self.info = INFO[config.DATA_FLAG]
        DataClass = getattr(medmnist, self.info['python_class'])
        
        # Base transform: ToTensor -> Normalize
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load the underlying MedMNIST dataset
        self.dataset = DataClass(
            split=split, 
            transform=self.base_transform, 
            download=config.DOWNLOAD_DATA,
            size=config.IMAGE_SIZE
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # clean_img is already a Tensor in [-1, 1] due to base_transform
        clean_img, _ = self.dataset[idx] 
        
        # Add simulated noise
        noisy_img = self.image_processor.add_noise(clean_img)
        
        return noisy_img, clean_img

class DataLoaderManager:
    def __init__(self, config, image_processor):
        self.config = config
        self.image_processor = image_processor
        self.info = INFO[config.DATA_FLAG]
        
    def get_dataloaders(self):
        train_dataset = NoisyTissueMNIST('train', self.config, self.image_processor)
        val_dataset = NoisyTissueMNIST('val', self.config, self.image_processor)
        test_dataset = NoisyTissueMNIST('test', self.config, self.image_processor)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS
        )
        
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        return train_loader, val_loader, test_loader

    def get_info(self):
        return self.info
