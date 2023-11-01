import gc
import math
import sys
import torch
from src.points import middle_points
from src.test_func import gramm_const
from src.pinn_core import *
from src.exact import *

class Loss:
    def __init__(self, epsilon: float, n_points_x: int, n_points_t: int, n_test_x: int, n_test_y: int, device: torch.device):
       self.epsilon = epsilon
       self.n_points_x = n_points_x
       self.n_points_t = n_points_t
       self.n_test_x = n_test_x
       self.n_test_y = n_test_y
       self.G = gramm_const(epsilon, n_test_x, n_points_t, device)
       self.x, self.t = middle_points((0,1), (0,1), n_points_x, n_points_t, True, device)

    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        t = self.t
        epsilon = self.epsilon
        device = x.device
        dx = 1.0 / x.numel()
        dt = 1.0 / t.numel()

        # norm is (v,v)_VM = epsilon (dv/dx,dvdx)+epsilon (dv/dy,dvdy)

        final_loss = torch.tensor(0.0)
        index = 0
        dpinn_dt = dfdt(pinn, x, t, order=1)
        dpinn_dx = dfdx(pinn, x, t, order=1)
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
                    + epsilon * dpinn_dx * test_dx *  dx * dt \
                    + dpinn_dx * test * dx * dt
                final_loss += loss_tmp_weak.sum().pow(2) * self.G[index]
                index += 1

        return final_loss

class Error:
    def __init__(self, epsilon: float, n_points_x: int, n_points_t: int, device: torch.device):
       self.epsilon = epsilon
       self.n_points_x = n_points_x
       self.n_points_t = n_points_t
       self.x, self.t = middle_points((0,1), (0,1), n_points_x, n_points_t, True, device)
       self.l2_exact_norm = self._l2_exact_norm()
       self.vm_exact_norm = self._vm_exact_norm()

    def l2_norm(self, pinn: PINN) -> float:
        size = self.x.numel
        diff = f(pinn, self.x, self.t)-exact_solution(self.x, self.t, self.epsilon)
        l2_z_norm = diff.pow(2).sum()/size
        l2_norm = math.sqrt(l2_z_norm) / self.l2_exact_norm
        return l2_norm
    
    def _l2_exact_norm(self) -> float:
        size = self.x.numel
        exact = exact_solution(self.x, self.t)
        l2_exact_norm = exact.pow(2).sum()/size
        return math.sqrt(l2_exact_norm)

    def vm_norm(self, pinn: PINN) -> float:
        epsilon = self.epsilon
        x = self.x
        t = self.t
        size = x.numel
 
        dz_dx = dfdx(pinn, x, t, order=1)
        exact_dx = exact_solution_dx(x, t, epsilon)
        diff_dx = dz_dx - exact_dx
        diff_dx_int = diff_dx.pow(2).sum()/size

        dz_dt = dfdt(pinn, x, t, order=1)
        exact_dt = exact_solution_dt(x,t)
        diff_dt = dz_dt - exact_dt
        diff_dt_int = diff_dt.pow(2).sum()/size

        vm_z_norm = epsilon*(diff_dx_int + diff_dt_int)

        vm_norm = math.sqrt(vm_z_norm) / self.vm_exact_norm
        return vm_norm

    def _vm_exact_norm(self) -> float:
        size = self.x.numel
        exact_dx = exact_solution_dx(self.x, self.t, self.epsilon)
        exact_dx_norm = exact_dx.pow(2).sum()/size
        exact_dt = exact_solution_dt(self.x, self.t, self.epsilon)
        exact_dt_norm = exact_dt.pow(2).sum()/size
        vm_exact_norm = self.epsilon*(exact_dx_norm + exact_dt_norm)
        return math.sqrt(vm_exact_norm)
