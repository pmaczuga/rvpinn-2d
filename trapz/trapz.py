import os
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import shutil

from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from params import *
from src.pinn_core import *
from src.loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


hx = 1/(N_POINTS_X)
ht = 1/(N_POINTS_T)

x_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]

#midpoints
x_raw = torch.linspace(x_domain[0]+0.5*hx, x_domain[1]-0.5*hx, steps=N_POINTS_X, requires_grad=True)
t_raw = torch.linspace(t_domain[0]+0.5*ht, t_domain[1]-0.5*ht, steps=N_POINTS_T, requires_grad=True)
grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

x = grids[0].flatten().reshape(-1, 1).to(device)
t = grids[1].flatten().reshape(-1, 1).to(device)

x_init_raw = torch.linspace(0.0, 1.0, steps=N_POINTS_INIT)
x_init_raw = x_init_raw*LENGTH
grids = torch.meshgrid(x_init_raw, t_raw, indexing="ij")
x_init = grids[0].flatten().reshape(-1, 1).to(device)


def sin_act(x):
    return torch.sin(x)

# pinn = PINN(LAYERS, NEURONS_PER_LAYER, pinning=False, act=nn.Tanh()).to(device)
pinn = PINN(LAYERS, NEURONS_PER_LAYER, pinning=True, act=nn.Tanh()).to(device)
# assert check_gradient(nn_approximator, x, t)

compute_loss(pinn, x=x, x_init=x_init, t=t, epsilon=EPSILON, length=LENGTH, total_time=TOTAL_TIME)

# train the PINN
loss_fn = partial(compute_loss, x=x, t=t, x_init=x_init, weight_f=WEIGHT_INTERIOR, weight_i=WEIGHT_INITIAL, weight_b=WEIGHT_BOUNDARY, epsilon=EPSILON, length=LENGTH, total_time=TOTAL_TIME)

start_time = time.time()
pinn_trained, loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)
end_time = time.time()
exec_time = end_time = start_time
print(f"Training took: {exec_time} s")

# Create result directory if it doesn't exist
try:
    os.makedirs("results/data")
except OSError as error:
    pass

torch.save(pinn_trained, "results/data/pinn.pt")
torch.save(loss_values, "results/data/loss_values.pt")
torch.save(exec_time, "results/data/exec_time.pt")
shutil.copy("params.py", "results/result.txt")
with open("results/result.txt", "a") as file:
        file.write(f"\n")
        file.write(f"\nTime = {exec_time}\n")
