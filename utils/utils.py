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
from aggregator import *

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

    for client_id, client_dir in enumerate(tqdm(os.listdir(root_path))):
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

def get_aggregator(
        aggregator_type,
        clients,
        global_learners_ensemble,
        lr,
        lr_lambda,
        mu,
        communication_probability,
        q,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None
):
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )

    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )

    elif aggregator_type == "decentralized":
        n_clients = len(clients)
        mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)

        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            mixing_matrix=mixing_matrix,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
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


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def get_communication_graph(n, p, seed):
    return nx.generators.random_graphs.binomial_graph(n=n, p=p, seed=seed)


def compute_mixing_matrix(adjacency_matrix):
    """
    computes the mixing matrix associated to a graph defined by its `adjacency_matrix` using
    FMMC (Fast Mixin Markov Chain), see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf

    :param adjacency_matrix: np.array()
    :return: optimal mixing matrix as np.array()
    """
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    constraints = [
        W == W.T,
        W @ np.ones((N, 1)) == np.ones((N, 1)),
        cp.multiply(W, network_mask) == np.zeros((N, N)),
        -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
        W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N),
        np.zeros((N, N)) <= W
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    mixing_matrix = W.value

    mixing_matrix *= adjacency_matrix

    mixing_matrix = np.multiply(mixing_matrix, mixing_matrix >= 0)

    # Force symmetry (added for numerical stability)
    for i in range(N):
        if np.abs(np.sum(mixing_matrix[i, i:])) >= 1e-20:
            mixing_matrix[i, i:] *= (1 - np.sum(mixing_matrix[i, :i])) / np.sum(mixing_matrix[i, i:])
            mixing_matrix[i:, i] = mixing_matrix[i, i:]

    return mixing_matrix


def get_mixing_matrix(n, p, seed):
    graph = get_communication_graph(n, p, seed)
    adjacency_matrix = nx.adjacency_matrix(graph, weight=None).todense()

    return compute_mixing_matrix(adjacency_matrix)

