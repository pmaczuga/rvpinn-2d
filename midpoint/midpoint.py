import os
import shutil
import time

import torch

from torch import nn

from params import *
from src.pinn_core import *
from src.loss import Loss, Error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

def sin_act(x):
    return torch.sin(x)

pinn = PINN(LAYERS, NEURONS_PER_LAYER, pinning=True, act=nn.Tanh()).to(device)

loss_fn = Loss(epsilon=EPSILON, n_points_x=N_POINTS_X, n_points_t=N_POINTS_T, n_test_x=20, n_test_y=20, device=device)
error_calc = Error(epsilon=EPSILON, n_points_x=N_POINTS_X, n_points_t=N_POINTS_T, device=device)

start_time = time.time()
train_result = train_model(pinn, loss_fn=loss_fn, error_calc=error_calc, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)
end_time = time.time()
exec_time = end_time - start_time
print(f"Training took: {exec_time} s")

# Create result directory if it doesn't exist
try:
    os.makedirs("results/data")
except OSError as error:
    pass

torch.save(pinn, "results/data/pinn.pt")
torch.save(train_result, "results/data/train_result.pt")
torch.save(exec_time, "results/data/exec_time.pt")
shutil.copy("params.py", "results/result.txt")
with open("results/result.txt", "a") as file:
        file.write(f"\n")
        file.write(f"\nTime = {exec_time}\n")
