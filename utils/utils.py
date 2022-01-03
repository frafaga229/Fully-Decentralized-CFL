import os
import time
import torch
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from learners.learner import *
from models import *
from utils.metrics import *
from utils.optimizer import *


def get_loaders(root_path, batch_size):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param root_path: path to the data folder
    :param batch_size:
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """

    inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, client_dir in enumerate(tqdm(os.listdir(root_path))):
        client_data_path = os.path.join(root_path, client_dir)

        train_dataset = TabularDataset(os.path.join(client_data_path, 'train.pkl'))
        train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TabularDataset(os.path.join(client_data_path, 'val.pkl'))
        val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TabularDataset(os.path.join(client_data_path, 'test.pkl'))
        test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators

def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: Learner

    """
    torch.manual_seed(seed)

    if name == "synthetic":
        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
            metric = binary_accuracy
            model = LinearModel(input_dim, 1).to(device)
            is_binary_classification = True
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            model = LinearModel(input_dim, output_dim).to(device)
            is_binary_classification = False

    else:
        raise NotImplementedError

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    return Learner(
        model=model,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        is_binary_classification=is_binary_classification
    )


class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    if args.decentralized:
        return f"{args.experiment}_decentralized"

    args_string = ""

    args_to_show = ["experiment", "method"]
    for arg in args_to_show:
        args_string = os.path.join(args_string, str(getattr(args, arg)))

    if args.locally_tune_clients:
        args_string += "_adapt"

    return args_string