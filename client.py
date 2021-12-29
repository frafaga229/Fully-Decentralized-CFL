import torch

class Client(object):

    def __init__(self, id, loader, model, criterion, device=None):

        self.id = id
        self.loader = loader
        self.model = model
        self.criterion = criterion
        if device is None:
            self.device = torch.device("cpu")
        self.device = device

    def local_step(self, n_epochs):
        """
        perform a local step
        """
        for _ in range(n_epochs):
            batch = next(iter(self.loader))
            self.compute_gradients(batch)
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            self.model.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)
            loss.backward()

        # how we push the gradients to other clients??


