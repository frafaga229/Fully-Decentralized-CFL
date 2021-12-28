
class Client(object):

    def __init__(
            self,
            id,
            dataset,
            model,
    ):

        self.id = id
        self.dataset = dataset
        self.model = model

    def local_step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform a local step on each node
        single_batch_flag: if true, the client only uses one batch to perform the update
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights
                )

        return client_updates