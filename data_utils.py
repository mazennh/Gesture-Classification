import splitfolders
import os
from typing import Tuple, List,Dict
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import warnings

warnings.filterwarnings("ignore")

def filter_data(input_path: str,
                output_path: str,
                split_path: str,
                classes_list: list,
                split_ratio: tuple = (0.8,0.1,0.1),
                seed: int = 42):
    """
    Filters specific classes from a dataset and splits them into train/val/test sets.

    This function copies the folders specified in `classes_list` from the `input_path`
    to the `output_path`. It then uses the `splitfolders` library to split these 
    collected classes into training, validation, and testing sets within `split_path`.

    Args:
        input_path (str): The root directory of the original dataset containing all class folders.
        output_path (str): Temporary directory to store the selected class folders before splitting.
                           (e.g., '/kaggle/working/temp_filtered')
        split_path (str): The final directory where the split data (train/val/test) will be saved.
                          (e.g., '/kaggle/working/final_dataset')
        split_ratio (tuple): A tuple of floats representing the split ratio. 
                             Example: (0.8, 0.1, 0.1) for 80% train, 10% val, 10% test.
        classes_list (list): A list of strings representing the names of the classes (folders) 
                             to extract and process.
        seed (int): Random seed for reproducibility of the split. Defaults to 42.

    Returns:
        None

    Raises:
        FileNotFoundError: If a class folder in `classes_list` does not exist in `input_path`.
    """
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Starting filtering process. Selecting {len(classes_list)} classes...")
    
    for cls in tqdm(classes_list, desc='Classes Processed'):
        src_path = os.path.join(input_path, cls)
        dst_path = os.path.join(output_path, cls)
        
        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True) 
        else:
            print(f"Warning: Class folder not found: {cls}")
            
    print("Copying complete. Starting dataset split...")
    
    splitfolders.ratio(output_path,
                       output=split_path, 
                       seed=seed,
                       ratio=split_ratio)
    
    print(f"Data split saved to: {split_path}")

def class_distribution(root_path: str):
    """
    Iterates through the root directory and prints the file count for each class folder.

    Args:
        root_path (str): The absolute path to the dataset directory containing subfolders.

    Returns:
        None: This function prints the output directly to the console.
    """

    print(f"{'CLASS NAME':<25} {'IMAGES'}")
    print("-" * 35)

    for folder in sorted(os.listdir(root_path)):
        folder_path = os.path.join(root_path, folder)
        
        if os.path.isdir(folder_path):
            count = len(os.listdir(folder_path))
            print(f"{folder:<25} {count}")


def create_dataloaders(
    data_dir: str, 
    batch_size: int =16, 
    img_size: int =224
    )->Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Creates PyTorch DataLoaders for Training, Validation, and Testing.

    This function applies data augmentation (RandomResizedCrop, Rotation, ColorJitter)
    to the training set, and standard resizing/normalization to the validation
    and test sets.

    Args:
        data_dir (str): Path to the root directory containing 'train', 'val', 
            and 'test' subdirectories.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.
        img_size (int, optional): The dimension to resize images to. Defaults to 224.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]: A tuple containing:
            - train_dataloader: DataLoader for the training set (shuffled).
            - val_dataloader: DataLoader for the validation set.
            - test_dataloader: DataLoader for the test set.
            - class_names: List of strings corresponding to the class labels.
            - class_to_idx: Dictionary mapping class names to their respective indices.
            
    Example:
        train_loader, val_loader, test_loader, classes, class_dict = create_dataloaders(
             data_dir="./data", 
             batch_size=32, 
             img_size=224)
    """
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() if os.cpu_count() else 2),
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() if os.cpu_count() else 2),
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() if os.cpu_count() else 2),
        pin_memory=True
    )

    img, _ = next(iter(train_dataloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print("="*3)
    print(f"Train data:\n{train_dataset}\nVal Data:\n{val_dataset}\nTest data:\n{test_dataset}")
    
    return (train_dataloader,
            val_dataloader,
            test_dataloader,
            train_dataset,
            train_dataset.classes,
            train_dataset.class_to_idx)