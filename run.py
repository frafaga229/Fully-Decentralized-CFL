from utils.utils import *
from utils.args import *
from utils.constants import *

from client import *

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, root_path, logs_root):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_root: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators =\
        get_loaders(
            root_path=root_path,
            batch_size=args_.bz
        )

    print("===> Initializing clients..")
    clients_ = []
    for client_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        learner =\
            get_learner(
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                input_dim=args_.input_dimension,
            )

        logs_path = os.path.join(logs_root, "client_{}".format(client_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = Client(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            id_=client_id
        )

        clients_.append(client)

    return clients_


def build_aggregator(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(
        args_,
        root_path=data_dir,
        logs_root=logs_dir
    )

    logs_path = os.path.join(logs_dir, "global")
    os.makedirs(logs_path, exist_ok=True)
    global_logger = SummaryWriter(logs_path)

    global_learner = \
        get_learner(
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            input_dim=args_.input_dimension
        )

    aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator_ = \
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learner=global_learner,
            sampling_rate=args_.sampling_rate,
            tol_1=args_.tol_1,
            tol_2=args_.tol_2,
            log_freq=args_.log_freq,
            global_logger=global_logger,
            verbose=args_.verbose,
            seed=args_.seed
        )

    return aggregator_


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    aggregator = build_aggregator(args)

    aggregator.write_logs()

    print("Training..")
    pbar = tqdm(total=args.n_rounds)
    current_round = 0
    while current_round <= args.n_rounds:

        aggregator.mix()

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round

    if "save_dir" in args:
        save_dir = os.path.join(args.save_path)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)
