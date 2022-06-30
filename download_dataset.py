from torchvision import datasets

def download_dataset():
    dataset_mnist_train = datasets.MNIST('./data', train=True, download=True, transform=None)
    dataset_mnist_test = datasets.MNIST('./data', train=False, download=True, transform=None)
    dataset_cifar10_train = datasets.CIFAR10('./data', train=True, download=True, transform=None)
    dataset_cifar10_test = datasets.CIFAR10('./data', train=False, download=True, transform=None)
    dataset_cifar100_train = datasets.CIFAR100('./data', train=True, download=True, transform=None)
    dataset_cifar100_test = datasets.CIFAR100('./data', train=False, download=True, transform=None)
    

if __name__ == "__main__":
    download_dataset()

