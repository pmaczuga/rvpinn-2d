import math
import numpy as np
import torch
            

def gramm_const(epsilon: float, n_test_x: int, n_test_y: int, device: torch.device) -> torch.Tensor:
    constants = torch.zeros(n_test_x * n_test_y)
    index = 0
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
            constants[index] = constant
            index += 1
    return constants
