import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class DeterministicHead(nn.Module):
    deterministic = True
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size=1024,
            num_layers=2,
            loss_coef=1.0,
            action_squash=False
    ):

        super().__init__()
        self.action_squash = action_squash
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if self.action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.loss_coef = loss_coef

    def forward(self, x):
        y = self.net(x)
        return y

    def get_action(self, x):
        return self.forward(x)

    def loss_fn(self, act, target, reduction="mean"):
        loss = F.mse_loss(act, target, reduction=reduction)
        return loss * self.loss_coef
