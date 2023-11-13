import math
import torch
from src.pinn_core import PINN, dfdt, dfdx
from src.points import middle_points
from src.test_func import gramm_const


class Loss:
    def __init__(self, epsilon: float, n_points_x: int, n_points_t: int, n_test_x: int, n_test_y: int, device: torch.device):
       self.epsilon = epsilon
       self.n_points_x = n_points_x
       self.n_points_t = n_points_t
       self.n_test_x = n_test_x
       self.n_test_y = n_test_y
       self.G = gramm_const(1.0, n_test_x, n_points_t, device).reshape(n_test_x*n_points_t)
       self.x, self.t = middle_points((0,1), (0,1), n_points_x, n_points_t, True, device)

    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        t = self.t
        epsilon = self.epsilon
        device = x.device
        dx = 1.0 / self.n_points_x
        dt = 1.0 / self.n_points_t

        final_loss = torch.tensor(0.0)
        index = 0
        dpinn_dt = dfdt(pinn, x, t, order=1)
        dpinn_dx = dfdx(pinn, x, t, order=1)
        rhs = self._rhs(x, t)
        for i in range(0, self.n_test_x):
            n = i + 1
            test_x = torch.sin(n*math.pi*x)
            test_x_dx = n * math.pi * torch.cos(n*math.pi*x)
            for j in range(0, self.n_points_t):
                m = j + 1
                test_t = torch.sin(m*math.pi*t)
                test_t_dt = m * math.pi * torch.cos(m*math.pi*t)
                test_dt = test_x.mul(test_t_dt)
                test_dx = test_t.mul(test_x_dx)
                test = test_x.mul(test_t)
                loss_tmp_weak = \
                    + epsilon * dpinn_dt * test_dt * dx * dt \
                    + epsilon * dpinn_dx * test_dx * dx * dt \
                    - rhs * test * dx * dt
                final_loss += loss_tmp_weak.sum().pow(2) * self.G[index]
                index += 1

        return final_loss
    
    def _rhs(self, x, y):
        return -4*torch.pi**2*torch.exp(torch.pi*(x-2*y))*torch.sin(torch.pi*(x-2*y))
