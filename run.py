import torch
import networkx as nx


def parse_args():


def main():
    args = parse_args()

    model = LinearModel()
    criterion = nn.CrossEntropyLoss()
    metric = accuracy
    device = torch.device("cpu")

    sizes = [15, 10]
    probs = [[0.6, 0.1], [0.1, 0.4]]
    graph = nx.stochastic_block_model(sizes, probs, seed=0)


if __name__ == "__main__":
    main()
