import math
import numpy as np
import torch
            

def gramm_const(epsilon: float, n_test_x: int, n_test_y: int, device: torch.device) -> torch.Tensor:
    constants = torch.zeros(n_test_x, n_test_y)
    for i in range(n_test_x):
        n = i + 1
        constant_1b = (2.0*math.pi*n-math.sin(2.0*math.pi*n))/(4.0*math.pi*n)
        constant_2a = epsilon * n*n*math.pi*math.pi *(2.0*math.pi*n+math.sin(2.0*math.pi*n))/(4.0*math.pi*n)
        for j in range(n_test_y):
            m = j + 1
            constant_1a = epsilon*m*m*math.pi*math.pi*(2.0*math.pi*m + math.sin(2.0*math.pi*m))/(4.0*math.pi*m)
            constant_2b = (2.0*math.pi*m - math.sin(2.0*math.pi*m))/(4.0*math.pi*m)
            constant_full = constant_1a*constant_1b + constant_2a*constant_2b
            constant = 1.0 / constant_full
            constants[i, j] = constant
    return constants

def test_x(x: torch.Tensor, N: int) -> torch.Tensor:
    x = x.reshape(-1, 1)
    n = torch.arange(1, N+1).to(x.device)
    return torch.sin(x*n*torch.pi)

def test_t(t:torch.Tensor, N: int) -> torch.Tensor:
    t = t.reshape(-1, 1)
    n = torch.arange(1, N+1).to(t.device)
    return torch.sin(t*n*torch.pi)

def test_x_dx(x: torch.Tensor, N: int) -> torch.Tensor:
    x = x.reshape(-1, 1)
    n = torch.arange(1, N+1).to(x.device)
    return torch.cos(n*x*torch.pi) * n * torch.pi

def test_t_dt(t: torch.Tensor, N: int) -> torch.Tensor:
    t = t.reshape(-1, 1)
    n = torch.arange(1, N+1).to(t.device)
    return torch.cos(n*t*torch.pi) * n * torch.pi
