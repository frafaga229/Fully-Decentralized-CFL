import os
import argparse

import networkx as nx

from utils import *


BOX = (-1.0, 1.0)

PATH = "all_data/"
METADATA_PATH = "metadata.json"
GRAPH_PATH = "graph.json"


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_clients',
        help='number of tasks/clients;',
        required=True,
        type=int
    )
    parser.add_argument(
        '--dimension',
        help='data dimension;',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--n_clusters',
        help='number of clusters;',
        default=2,
        type=int
    )
    parser.add_argument(
        '--graph_hetero_level',
        help='Heterogeneity level of the graph clustering; smaller value for clearer clustering',
        type=float,
        default=0.
    )
    parser.add_argument(
        '--noise_level',
        help='Noise level;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--data_hetero_level',
        help='Heterogeneity level of the models of every cluster;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--n_train',
        help='size of train set;',
        type=int,
        default=1_000
    )
    parser.add_argument(
        '--n_val',
        help='size of validation set;',
        type=int,
        default=1_000
    )
    parser.add_argument(
        '--n_test',
        help='size of test set;',
        type=int,
        default=1_000
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=12345,
        required=False
    )

    if args_list is None:
        return parser.parse_args()

    else:
        return parser.parse_args(args_list)


if __name__ == "__main__":
    os.makedirs(PATH, exist_ok=True)

    args = parse_args()

    args.graph_hetero_level = max(args.graph_hetero_level, (args.n_clusters-1)/args.n_clusters)

    data_generator = \
        SyntheticDataGenerator(
            n_clusters=args.n_clusters,
            dim=args.dimension,
            n_train_samples=args.n_train,
            n_val_samples=args.n_val,
            n_test_samples=args.n_test,
            clusters_hetero=args.data_hetero_level,
            noise_level=args.noise_level,
            seed=args.seed
        )

    print("\n==> Save metadata..")
    data_generator.save_metadata(json_path=METADATA_PATH)

    clients_per_cluster = iid_divide(list(range(args.n_clients)), args.n_clusters)
    n_clients_per_cluster = [len(l) for l in clients_per_cluster]

    prob_matrix = np.ones((args.n_clusters, args.n_clusters)) - np.eye(args.n_clusters)
    prob_matrix *= (args.graph_hetero_level / (args.n_clusters - 1))
    prob_matrix += np.diag((1-args.graph_hetero_level)*np.ones(args.n_clusters))

    # TODO
    # print("\n==>Save graph..")
    # graph = nx.stochastic_block_model(n_clients_per_clusters, prob_matrix, seed=args.seed)
    # json_graph = nx.readwrite.json_graph.node_link_data(graph)

    # with open(GRAPH_PATH, "w") as f:
    #    json.dump(json_graph, f)

    print("\n==>Save data..")

    all_data = data_generator.generate_data()

    for mode in ["train", "val", "test"]:
        for cluster_id in range(args.n_clusters):
            cluster_data = all_data[mode][cluster_id]

            data_per_client = {
                "x": iid_divide(cluster_data["x"], n_clients_per_cluster[cluster_id]),
                "y": iid_divide(cluster_data["y"], n_clients_per_cluster[cluster_id])
            }

            for idx, client_id in enumerate(clients_per_cluster[cluster_id]):
                client_data = {
                    "x": data_per_client["x"][idx],
                    "y": data_per_client["y"][idx]
                }

                client_dir = os.path.join(PATH, f"{client_id}")
                os.makedirs(client_dir, exist_ok=True)

                save_path = os.path.join(client_dir, f"{mode}.json")

                with open(save_path, "w") as f:
                    json.dump(client_data, f)
