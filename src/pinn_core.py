import math
from typing import Callable
import numpy as np
import torch
from torch import nn

def shift(x, t) -> torch.Tensor:
  shift_x = torch.sin(math.pi*x)
  shift_t =(1.0-t)
  res = shift_x.mul(shift_t)
  return res

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False):

        super().__init__()

        self.pinning = pinning

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):

        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        # if requested pin the boundary conditions
        # using a surrogate model: (x - 0) * (x - L) * NN(x)
        if self.pinning:
            logits *= (x - 0.0)*(x - 1.0)*(t - 0.0)*(t - 1.0)

        return logits

def f(pinn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, t)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the time variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, x, order=order)


class TrainResult:
    def __init__(
            self,
            loss: np.ndarray,
            vm_norm: np.ndarray,
            l2_norm: np.ndarray,
            vm_exact_norm: float,
            l2_exact_norm: float
        ):
        self.loss = loss
        self.vm_norm = vm_norm
        self.l2_norm = l2_norm
        self.vm_exact_norm = vm_exact_norm
        self.l2_exact_norm = l2_exact_norm


def train_model(
    pinn: PINN,
    loss_fn: Callable,
    error_calc,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
    device="cpu"
) -> TrainResult:

    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)
    loss_values = []
    l2_norm = []
    vm_norm = []
    for epoch in range(max_epochs):

        try:

            loss = loss_fn(pinn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            l2_norm.append(error_calc.l2_norm(pinn))
            vm_norm.append(error_calc.vm_norm(pinn))
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    result = TrainResult(
        loss          = np.array(loss_values), 
        vm_norm       = np.array(l2_norm), 
        l2_norm       = np.array(vm_norm), 
        vm_exact_norm = error_calc.vm_exact_norm,
        l2_exact_norm = error_calc.l2_exact_norm
    )
    
    return result
