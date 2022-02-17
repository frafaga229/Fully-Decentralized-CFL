import time
import pickle
import numpy as np
from tqdm import tqdm

import numpy.linalg as LA


BOX = (-1.0, 1.0)


class SyntheticDataGenerator:
    r"""
    Synthetic data simulator.


    Attributes
    ----------

    Methods
    -------
    generate_cluster_data

    save_data(dir_path)

    save_metadata(path_)

    """
    def __init__(
            self,
            n_clusters,
            dim,
            n_train_samples,
            n_val_samples,
            n_test_samples,
            clusters_hetero=0.,
            noise_level=0.,
            seed=None
    ):

        self.n_clusters = n_clusters
        self.dim = dim

        self.clusters_hetero = clusters_hetero
        self.noise_level = noise_level

        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples

        self.seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = np.random.default_rng(self.seed)

        self.base_model = self.rng.normal(size=self.dim)

        self.clusters_models = \
            self.rng.normal(
                loc=self.base_model,
                scale=self.clusters_hetero,
                size=(self.n_clusters, self.dim)
            )

    def generate_cluster_data(self, cluster_id, n_samples):
        x = self.rng.uniform(BOX[0], BOX[1], size=(n_samples, self.dim))
        y = x @ (self.clusters_models[cluster_id] / LA.norm(self.clusters_models[cluster_id]))
        y += self.rng.normal(scale=self.noise_level, size=y.shape)

        return x.tolist(), y.tolist()

    def generate_data(self):
        """
        generate data per cluster

        """
        all_data = dict()
        for mode, n_samples in \
                zip(["train", "val", "test"], [self.n_train_samples, self.n_val_samples, self.n_test_samples]):

            print(f"\n===> Generating {mode} data..")

            data_per_cluster = dict()
            for cluster_id in tqdm(range(self.n_clusters)):

                x, y = self.generate_cluster_data(
                    cluster_id=cluster_id,
                    n_samples=n_samples
                )

                data_per_cluster[cluster_id] = {"x": x, "y": y}

            all_data[mode] = data_per_cluster

        return all_data

    def save_metadata(self, path_):
        metadata = {
            "base_model": self.base_model.tolist(),
            "clusters_models": self.clusters_models.tolist()
        }
        with open(path_, 'wb') as f:
            pickle.dump(metadata, f)


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def save_data(x, y, path_):
    data = list(zip(x, y))
    with open(path_, 'wb') as f:
        pickle.dump(data, f)