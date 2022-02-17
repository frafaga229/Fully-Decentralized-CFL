from abc import ABC, abstractmethod

import os
import time
import random

import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from copy import deepcopy
from utils.torch_utils import *


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


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: List[Client]

    global_learner: List[Learner]

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients:

    n_learners:

    clients_weights:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_logger:

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__
    mix

    update_clients

    write_logs

    save_state

    load_state

    """

    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            verbose=0,
            seed=None,
            *args,
            **kwargs
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.clients = clients

        self.global_learner = global_learner
        self.device = self.global_learner.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_logger = global_logger

        self.model_dim = self.global_learner.model_dim

        self.n_clients = len(clients)

        self.clients_weights = \
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def write_logs(self):
        for global_logger, clients in [
            (self.global_logger, self.clients)
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_metric, val_loss, val_metric, test_loss, test_metric = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    print(f"Train Loss: {train_loss:.3f} |", end="")
                    print(f"Test Loss: {test_loss:.3f}")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_metric * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_metric * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} |", end="")
                print(f"Test Loss: {global_test_loss:.3f}")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        save_path = os.path.join(dir_path, f"chkpts.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        chkpts_path = os.path.join(dir_path, f"chkpts.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=self.n_clients_per_round)


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        learners = [client.learner for client in self.clients]
        average_learners(learners, self.global_learner, weights=self.clients_weights)

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:
            copy_model(client.learner.model, self.global_learner.model)


class ClusteredAggregator(Aggregator):
    """
    # TODO: check this class
    Implements
     `Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints`.

     Follows implementation from https://github.com/felisat/clustered-federated-learning
    """

    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            verbose=0,
            tol_1=0.4,
            tol_2=1.6,
            seed=None
    ):

        super(ClusteredAggregator, self).__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_logger=global_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            verbose=verbose,
            seed=seed
        )

        assert self.sampling_rate == 1.0, f"`sampling_rate` is {sampling_rate}, should be {1.0}," \
                                          f" ClusteredAggregator only supports full clients participation."

        self.tol_1 = tol_1
        self.tol_2 = tol_2

        self.global_learners = [self.global_learner]
        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        self.n_clusters = 1

    def mix(self):
        clients_updates = np.zeros((self.n_clients, self.model_dim))

        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step()

        similarities = pairwise_distances(clients_updates, metric="cosine")

        new_cluster_indices = []
        for indices in self.clusters_indices:
            max_update_norm = LA.norm(clients_updates[indices], axis=1).max()
            mean_update_norm = LA.norm(np.mean(clients_updates[indices], axis=0))

            if mean_update_norm < self.tol_1 and max_update_norm > self.tol_2 and len(indices) > 2:
                clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
                clustering.fit(similarities[indices][:, indices])
                cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
                cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
                new_cluster_indices += [cluster_1, cluster_2]
            else:
                new_cluster_indices += [indices]

        self.clusters_indices = new_cluster_indices

        self.n_clusters = len(self.clusters_indices)

        self.global_learners = [deepcopy(self.clients[0].learner) for _ in range(self.n_clusters)]

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_clients = [self.clients[i] for i in indices]

            average_learners(
                learners=[client.learner for client in cluster_clients],
                target_learner=self.global_learners[cluster_id],
                weights=self.clients_weights[indices] / self.clients_weights[indices].sum()
            )

        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_learners = self.global_learners[cluster_id]

            for i in indices:
                copy_model(
                    target=self.clients[i].learner.model,
                    source=cluster_learners.model
                )


class DecentralizedAggregator(Aggregator):
    def __init__(
            self,
            clients,
            global_learner,
            mixing_matrix,
            log_freq,
            global_logger,
            sampling_rate=1.,
            sample_with_replacement=True,
            verbose=0,
            seed=None):

        super(DecentralizedAggregator, self).__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_logger=global_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            verbose=verbose,
            seed=seed
        )

        self.mixing_matrix = mixing_matrix
        assert self.sampling_rate >= 1, "partial sampling is not supported with DecentralizedAggregator"

    def update_clients(self):
        pass

    def mix(self):
        # update local models
        for client in self.clients:
            client.step()

        # mix models
        mixing_matrix = torch.tensor(
            self.mixing_matrix.copy(),
            dtype=torch.float32,
            device=self.device
        )

        state_dicts = [client.learner.model.state_dict() for client in self.clients]

        for key, param in self.global_learner.model.state_dict().items():
            shape_ = param.shape
            models_params = torch.zeros(self.n_clients, int(np.prod(shape_)), device=self.device)

            for ii, sd in enumerate(state_dicts):
                models_params[ii] = sd[key].view(1, -1)

            models_params = mixing_matrix @ models_params

            for ii, sd in enumerate(state_dicts):
                sd[key] = models_params[ii].view(shape_)

        for client_id, client in enumerate(self.clients):
            client.learner.model.load_state_dict(state_dicts[client_id])

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
