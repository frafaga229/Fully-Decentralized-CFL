import torch
from learners.learner import *
from copy import deepcopy
from utils.utils import copy_model


class Client(object):
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False
    ):

        self.learner = learner
        self.n_learners = 1
        self.tune_locally = tune_locally

        if self.tune_locally:
            self.tuned_learner = deepcopy(self.learner)
        else:
            self.tuned_learner = None

        self.binary_classification_flag = self.learner.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.local_steps = local_steps

        self.counter = 0
        self.logger = logger

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch

    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learner.fit_batch(
                    batch=batch,
                )
        else:
            client_updates = \
                self.learner.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps
                )

        return client_updates

    def write_logs(self):

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learner.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learner.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learner.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.learner.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc
