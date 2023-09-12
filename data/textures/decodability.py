import torch
from torch import nn


class TextureDecodingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextureDecodingModel, self).__init__()
        self.criterion = None
        self.optimizer = None
        self.metrics = None
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer(self.parameters())
        self.criterion = loss
        self.metrics = metrics

    def fit(self, X, y, epochs, validation_data=None):
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        self.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if validation_data:
                X_val, y_val = validation_data
                X_val = torch.Tensor(X_val)
                y_val = torch.Tensor(y_val)
                self.eval()
                y_pred_val = self.model(X_val)
                val_loss = self.criterion(y_pred_val, y_val)
                print(f'Epoch {epoch} train loss: {loss.item()} val loss: {val_loss.item()}')
            else:
                print(f'Epoch {epoch} train loss: {loss.item()}')

    def evaluate(self):
        pass