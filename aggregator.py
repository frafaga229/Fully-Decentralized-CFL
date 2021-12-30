from copy import deepcopy

import torch
import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering



class Aggregator:
    def __init__(self, clients, graph_manager):

        self.clients = clients
        self.global_learners = [self.global_learners_ensemble]
        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        self.graph_manager = graph_manager

    def mix(self):
        clients_updates = np.zeros((self.n_clients, self.model_dim))

        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.local_step()

        similarities = np.zeros((self.n_clients, self.n_clients))

        for learner_id in range(self.n_learners):
            similarities[learner_id] = pairwise_distances(clients_updates[:, learner_id, :], metric="cosine")

        similarities = similarities.mean(axis=0)

        # self.mixing_matrix = mixing_matrix
        #
        #
        # for learner_id, global_learner in enumerate(self.global_learners_ensemble):
        #     state_dicts = [client.learners_ensemble[learner_id].model.state_dict() for client in self.clients]
        #
        #     for key, param in global_learner.model.state_dict().items():
        #         shape_ = param.shape
        #         models_params = torch.zeros(self.n_clusters, int(np.prod(shape_)), device=self.device)
        #
        #         for ii, sd in enumerate(state_dicts):
        #             models_params[ii] = sd[key].view(1, -1)
        #
        #         models_params = mixing_matrix @ models_params
        #
        #         for ii, sd in enumerate(state_dicts):
        #             sd[key] = models_params[ii].view(shape_)
        #
        #     for client_id, client in enumerate(self.clients):
        #         client.learners_ensemble[learner_id].model.load_state_dict(state_dicts[client_id])
        #
        # self.c_round += 1
