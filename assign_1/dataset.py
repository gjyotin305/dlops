import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define transformation for datasets
transform = v2.Compose([
    v2.ToTensor(),
    # For MNIST, normalize grayscale images
    v2.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True)
mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True)

# Modify transform for CIFAR datasets
cifar_transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB images
])
# Load CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=cifar_transform,
    download=True)
cifar10_test = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=cifar_transform,
    download=True)

# Load CIFAR-100 dataset
cifar100_train = datasets.CIFAR100(
    root="./data",
    train=True,
    transform=cifar_transform,
    download=True)
cifar100_test = datasets.CIFAR100(
    root="./data",
    train=False,
    transform=cifar_transform,
    download=True)

# Create data loaders
batch_size = 64
mnist_train_loader = DataLoader(
    mnist_train,
    batch_size=batch_size,
    shuffle=True
)
mnist_test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False)
cifar10_train_loader = DataLoader(
    cifar10_train,
    batch_size=batch_size,
    shuffle=True)
cifar10_test_loader = DataLoader(
    cifar10_test,
    batch_size=batch_size,
    shuffle=False)
cifar100_train_loader = DataLoader(
    cifar100_train,
    batch_size=batch_size,
    shuffle=True)
cifar100_test_loader = DataLoader(
    cifar100_test,
    batch_size=batch_size,
    shuffle=False)
