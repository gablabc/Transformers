

import wandb
import torch
import numpy.random
from math import ceil
import random
import torchsummary


# Local imports
from utils import Wandb_Config, Data_Config
from utils import get_mnist_dataloaders
from models import TransformerModel
from methods import method_reduce_lr

if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    ################################### Setup #################################
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Wandb_Config, "wandb")
    parser.add_arguments(Data_Config, "data")

    TransformerModel.add_argparse_args(parser)
    method_reduce_lr.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    print(args)

    # Fix seed for initialisation for complete reproducibility
    #torch.use_deterministic_algorithms(True)
    random.seed(args.method.seed)
    torch.manual_seed(args.method.seed)
    numpy.random.seed(seed=args.method.seed)

    # Init W&B run
    if args.wandb.wandb:
        wandb.init(project=args.wandb.wandb_project, entity=args.wandb.wandb_entity)
        wandb.run.name = f"{args.data.name} - {wandb.run.name}"
        for main_arg in vars(args):
            if not main_arg == "wandb":
                wandb.config.update(getattr(args, main_arg).__dict__)
                
    # Number of times the user wants to log the valid perf
    n_logs = ceil(args.method.n_epochs / args.method.n_epochs_log)

    # Load the dataset and process is for MLPs
    load_train, load_valid, load_test, _, _, _ = get_mnist_dataloaders(batch_size=args.data.batch_size)
    model = TransformerModel.from_argparse_args(args)
        
    print(model)
    torchsummary.summary(model)

    # Generate fit method
    method = method_reduce_lr.from_argparse_args(args)

    # Train
    for log in range(n_logs):
        # Apply the method for n_epochs_log and then log the valid perf
        # This allows to avoid refitting models with different n_epochs
        method.apply(model, load_train, wandb=args.wandb.wandb, fresh=not bool(log))

        # Assess performance
        #models.eval()
        # perfs = accuracy_and_loss(load_valid, model, mask, device)


    # # Summaries on W&B
    # if args.wandb.wandb:
    #     # Put average error of the aggregated predictor h for each log
    #     for log in range(n_logs):
    #         #perf_metric = "RMSE" if models.hparams.task == "regression" else "Error Rate" 
    #         wandb.run.summary[f"3th_quartile_valid_loss_{log}"] = torch.mean(perfs[:, log])
    #     wandb.finish()
