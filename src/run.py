import os
import shutil
import time

import torch

from torch import nn
from src.exact import ExpSinsExactSolution, SinsExactSolution

from src.params import Params
from src.pinn_core import *
from src.loss import Loss, Error

def run(p: Params, device: torch.device):
    print(f"Running on: {device}")
    
    if (p.equation == "sins"):
        exact = SinsExactSolution()
    elif (p.equation == "exp-sins"):
        exact = ExpSinsExactSolution()

    pinn = PINN(p.layers, p.neurons_per_layer, pinning=True, act=nn.Tanh()).to(device)

    loss_fn = Loss(epsilon=p.epsilon, n_points_x=p.n_points_x, n_points_t=p.n_points_t, n_test_x=p.n_test_x, n_test_t=p.n_test_t, device=device)
    # old_loss_fn = src.old_loss.Loss(epsilon=EPSILON, n_points_x=N_POINTS_X, n_points_t=N_POINTS_T, n_test_x=N_TEST_X, n_test_y=N_TEST_T, device=device)
    error_calc = Error(epsilon=p.epsilon, n_points_x=p.n_points_x_error, n_points_t=p.n_points_t_error, device=device)

    start_time = time.time()
    train_result = train_model(pinn, loss_fn=loss_fn, error_calc=error_calc, learning_rate=p.learning_rate, max_epochs=p.epochs)
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Training took: {exec_time} s")

    # Create result directory if it doesn't exist
    try:
        os.makedirs(f"results/{p.tag}/data")
    except OSError as error:
        pass

    torch.save(pinn, f"results/{p.tag}/data/pinn.pt")
    torch.save(train_result, f"results/{p.tag}/data/train_result.pt")
    torch.save(exec_time, f"results/{p.tag}/data/exec_time.pt")
    shutil.copy("params.ini", f"results/{p.tag}/params.ini")
    with open(f"results/{p.tag}/result.txt", "w") as file:
        file.write(f"\nTime = {exec_time}\n")
