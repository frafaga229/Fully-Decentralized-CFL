class Client(object):
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            id_=-1
    ):

        self.id = id_

        self.learner = learner
        self.n_learners = 1

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

    def step(self, single_batch_flag=False):
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

        train_loss, train_metric = self.learner.evaluate_iterator(self.train_iterator)
        val_loss, val_metric = self.learner.evaluate_iterator(self.val_iterator)
        test_loss, test_metric = self.learner.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_metric, self.counter)
        self.logger.add_scalar("Val/Loss", val_loss, self.counter)
        self.logger.add_scalar("Val/Metric", val_metric, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_metric, self.counter)

        return train_loss, train_metric, val_loss, val_metric, test_loss, test_metric
