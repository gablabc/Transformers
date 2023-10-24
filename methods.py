""" Learning algorithms used to train Deep Ensembles """

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import trange
import timeit
import wandb
import random
import numpy

from typing import Optional
from dataclasses import dataclass
from simple_parsing import ArgumentParser


class method_reduce_lr(nn.Module):
    """ 
    Class that trains an Neural Network by minimizing the training set loss
    and reducing the learning_rate when stagnating for several epochs. 
    """

    @dataclass
    class HParams():
        seed: Optional[int] = 3   # Number of the seed used for training.
        n_epochs: int = 10       # Max. number of epochs to train the ensemble
        n_epochs_log: str = ""    # Interval of epochs where valid loss is logged. Default is n_epochs

        def __post_init__(self):  # Hack
            if type(self.n_epochs_log)==str:
                self.n_epochs_log: int = self.n_epochs if self.n_epochs_log == "" else int(self.n_epochs_log)

        learning_rate: float = 0.001  # Learning rate
        patience: int = 10            # Number of epochs before reducing the learning rate (if use_scheduler)
        lr_decay: float = 0.1         # Factor used for lr decay (if use_scheduler)
        use_scheduler: bool = True    # To use the reduce_lr_on_plateau scheduler


    def __init__(self, hparams: HParams=None, **kwargs):
        """Initialization of the method_reduce_lr class. The user can either give a premade hparams object 
        made from the Hparams class or give the keyword arguments to make one.

        Args:
            hparams (HParams, optional): HParams object to specify the models characteristics. Defaults to None.
        """
        self.hparams = hparams or self.HParams(**kwargs)
        super().__init__()


    @classmethod
    def from_argparse_args(cls, args):
        """Creates an instance of this Class from the parsed arguments."""
        hparams: cls.HParams = args.method
        return cls(hparams=hparams)


    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        """Adds command-line arguments for this Class to an argument parser."""
        parser.add_arguments(cls.HParams, "method")


    def one_epoch(self):
        """Trains the model for one epoch.

        Returns:
            dict: Dictionary that contains the mean loss value (of all (x,y) pairs) 
            and the learning rate at the end of the epoch.
        """
        self.scores = {'loss': 0., }
        example_count = 0.

        self.model.train(True)
        with torch.enable_grad():
            # x being all xs of a batch
            for (x, y) in self.train_loader:
                self.optimizer.zero_grad()

                L = self.model.loss(x, y)
                L.backward()

                lr = self.optimizer.param_groups[0]['lr']

                self.optimizer.step()

                self.scores['loss'] += L.item() * len(x)
                example_count += len(x)
                
                # Log on W&B in real-time
                if self.wandb:
                    wandb.log({'loss': L.item(), 'lr': lr})

        mean_scores = {'loss': self.scores['loss'] / example_count, 'lr': lr}
        
        return mean_scores


    def apply(self, model, train_loader, wandb=False, fresh=True):
        """Method used to train the Ensemble of models and log the loss every n_epochs_log epochs.

        Parameters
        ----------
        model (nn.Module)
            Pytorch model to train
        train_loader (DataLoader): 
            Loads the mini-batch of training examples.
        wandb (bool, optional):
            Set to True to log the run on Weights and Biases, False otherwise. Defaults to False.
        fresh (bool, optional):
            Set to True to start from a fresh set of parameters, False otherwise. Defaults to True.

        Returns
        --------
        float: Time it took to iterate through all the number of epochs where valid loss is logged (n_epochs_log).
        """

    
        # Init additionnal attributes
        self.wandb = wandb
        self.train_loader = train_loader
        self.model = model

        # If starting the procedure
        if fresh:
            # Set optimizer
            self.n_epochs_done = 0
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.hparams.learning_rate)
            if self.hparams.use_scheduler:
                self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                                   patience=self.hparams.patience, 
                                                   factor=self.hparams.lr_decay, 
                                                   min_lr=0)
        # Main training loop
        # self.logs = {'loss': [], "lr": []}
        start = timeit.default_timer()
        
        with trange(self.n_epochs_done, min(self.hparams.n_epochs,
                                            self.hparams.n_epochs_log + 
                                            self.n_epochs_done)) as tr:
            tr.set_description(desc=self.model.name, refresh=False)
            for _ in tr:
                # if self.n_epochs_done==34:
                #     print("bob")
                scores = self.one_epoch()
                if self.hparams.use_scheduler:
                    self.scheduler.step(scores['loss'])
                
                # Report
                tr.set_postfix(scores)
                # for key, values in logs.items():
                #     values.append(scores[key])

                #if scores['lr'] <= 1e-4:
                #    break
                self.n_epochs_done += 1
        
        print("\n")
        stop = timeit.default_timer()
        time = stop - start

        return time



# # Store trained model
# def store_model(model, best_accuracy_val, path):

#     # Write accuracy in txt file
#     file = open(f"{root}Experiments/{experiment}/{model.name}/hyperparams.txt", "w")
#     file.write(str(best_accuracy_val) + '\n')
#     file.write(str(kwargs))

#     # Store best model
#     torch.save(model.state_dict(), f"{root}Experiments/{experiment}/{model.name}/model.pt")

# # Load trained model
# def load_model(model):

#     # Read accuracy from txt file
#     #file = open(f"{root}Experiments/{experiment}/{model.name}/hyperparams.txt", "r")
#     #best_accuracy_val = float(file.readline())
#     best_accuracy_val = 0

#     model.load_state_dict(torch.load(f"{root}Experiments/{experiment}/{model.name}/model.pt"))

#     return best_accuracy_val



# def train(model, lr = 1e-4, nb_epochs = 10, batch_size = 32):

#     # Generate folders for this specific model name
#     if not os.path.exists(f"{root}Experiments/{experiment}/{model.name}"):
#         os.mkdir(f"{root}Experiments/{experiment}/{model.name}")

#     # Load the best accuracy as of yet for the current model and solver
#     if os.path.exists(f"{root}Experiments/{experiment}/{model.name}/model.pt"):
#         best_accuracy_val = load_model(model)
#         print(f"Loading model with val accuracy : {best_accuracy_val * 100: .2f}%")
#     else:
#         print("Best validation accuracy as of yet 0 %")
#         best_accuracy_val = 0

#     data_loader_train, data_loader_val, _, len_train, len_val, _ = get_mnist_dataloaders(0.1, batch_size = batch_size)

#     logger = Logger()
#     optim = torch.optim.Adam(model.parameters(), lr=lr)

#     loss_func = nn.CrossEntropyLoss()

#     if model.name == "Transformer":
#         mask = subsequent_mask(784).to(device)

#     for epoch in range(nb_epochs):
#         current_loss = 0
#         current_acc = 0
#         batch_idx = 1
#         for x, y in data_loader_train:

#             if batch_idx % 200 == 0:
#                 print(f"Train Epoch: {epoch + 1} [{batch_idx * batch_size} / {len_train}]")

#             x = x.to(device=device)
#             y = y.to(device=device)

#             if model.name == "Transformer":
#                 output = model(x, mask)
#             else:
#                 output = model(x)
#             loss = loss_func(output, y)

#             optim.zero_grad()
#             loss.backward()

#             nn.utils.clip_grad_value_(model.parameters(), 10)

#             optim.step()
#             batch_idx += 1


#         accuracy_train, loss_train = accuracy_and_loss_whole_dataset(data_loader_train, len_train, model)
#         accuracy_val, loss_val = accuracy_and_loss_whole_dataset(data_loader_val, len_val, model)

#         # Record best model
#         if accuracy_val > best_accuracy_val:
#             best_accuracy_val = accuracy_val
#             store_model(model, best_accuracy_val, lr = lr, batch_size = batch_size, epoch = epoch + 1)
#         logger.log(accuracy_train, loss_train, accuracy_val, loss_val)

#         print("\n" + \
#             f"Train: loss={loss_train:.3f}, accuracy={accuracy_train*100:.1f}%  \t" + \
#             f"Validation: loss={loss_val:.3f}, accuracy={ accuracy_val*100:.1f}% \n", flush=True)

#     return logger