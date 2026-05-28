import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision import datasets, transforms #datasets is responsible for downloading and loading our dataset, transform is responsible for appying transformation to our data like translation, rotataion, edc
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_cifar10_loaders(batch_size=8):
    """Builds the pipeline and returns the data loaders."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),#pixel value ranges from 0 to 1 and batch size w and h, image format is 0-255 and number of channels w and h,
        transforms.Normalize(0.5, 0.5)
    ])

    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    return train_loader, train_dataset

# --- THE QUARANTINE ZONE ---
# This only runs when you hit "Run" on this specific file!
if __name__ == "__main__":
    print("Testing Vision Pipeline...")

    # 1. Call the function you just built
    train_loader, train_dataset = get_cifar10_loaders()
    print(f"Total images: {len(train_dataset)}")

    # 2. Your awesome Matplotlib plotting logic
    for images, labels in train_loader:
        print(f"Batch Image size: {images.shape}, Labels size: {labels.shape}")

        images = images * 0.5 + 0.5  # Undo the normalization

        fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
        for i, img in enumerate(images):
            axes[i].imshow(img.permute(1, 2, 0))  # Convert CHW to HWC
            axes[i].axis("off")  # Turned axis off for cleaner image viewing!

        plt.show()
        break  # Only show one batch


