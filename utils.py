import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from dataclasses import dataclass

# Load the MNIST dataset
def get_mnist_dataloaders(val_percentage=0.3, batch_size=1):
    dataset = datasets.MNIST(f"dataset", train=True,  download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                       transforms.Lambda(lambda x: x.view(-1, 1))]))
    dataset_test = datasets.MNIST(f"dataset", train=False, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x.view(-1, 1))]))
    len_train = int(len(dataset) * (1 - val_percentage))
    len_val = len(dataset) - len_train
    dataset_train, dataset_val = random_split(dataset, [len_train, len_val])

    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=2)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader_train, data_loader_val, data_loader_test, \
                                len_train, len_val, len(dataset_test)


# # Go through the whole database and compute average loss and accuracy
# def accuracy_and_loss(data_loader, model, mask, device):
#     losses = 0
#     accuracies = 0
#     loss_func = nn.CrossEntropyLoss(reduction='sum')
#     model.eval()

#     with torch.no_grad():
#         for x, y in data_loader:

#             x = x.to(device=device)
#             y = y.to(device=device)

#             if model.name == "Transformer":
#                 output = model(x, mask)
#             else:
#                 output = model(x)

#             losses += loss_func(output, y).item()
#             accuracies += (y == torch.argmax(output, axis = 1)).sum().item()
#     model.train()
#     return accuracies /  len_data, losses /  len_data



@dataclass
class Wandb_Config:
    wandb: bool = False  # Use wandb logging
    wandb_project: str = "Transformers"  # Which wandb project to use
    wandb_entity: str = "galabc"  # Which wandb entity to use

@dataclass
class Data_Config:
    name: str = "MNIST"
    batch_size: int = 20  # Mini batch size
    scaler: str = "Standard"  # Scaler used for features and target

@dataclass
class Search_Config:
    n_splits: int = 5  # Number of train/valid splits
    split_type: str = "Shuffle" # Type of cross-valid "Shuffle" "K-fold"
    split_seed: int = 1 # Seed for the train/valid splits reproducability
