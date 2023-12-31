import gc
import math
import sys
import torch
from src.points import middle_points
from src.test_func import gramm_const, test_t, test_t_dt, test_x, test_x_dx
from src.pinn_core import *
from src.exact import *

class Loss:
    def __init__(self, 
                 epsilon: float, 
                 n_points_x: int, 
                 n_points_t: int, 
                 n_test_x: int, 
                 n_test_t: int, 
                 equation: str,
                 exact: ExactSolution,
                 device: torch.device):
       self.epsilon = epsilon
       self.n_points_x = n_points_x
       self.n_points_t = n_points_t
       self.n_test_x = n_test_x
       self.n_test_t = n_test_t
       self.G = gramm_const(1.0, n_test_x, n_test_t, device)
       self.x, self.t = middle_points((0,1), (0,1), n_points_x, n_points_t, True, device)
       self.n = torch.arange(1, self.n_test_x+1).to(device)
       self.m = torch.arange(1, self.n_test_t+1).to(device)
       self.exact = exact
       self.equation = equation

    def __call__(self, pinn: PINN) -> torch.Tensor:
        x = self.x
        t = self.t
        epsilon = self.epsilon
        device = x.device
        dx = 1.0 / self.n_points_x
        dt = 1.0 / self.n_points_t

        dpinn_dx = dfdx(pinn, x, t, order=1).reshape(self.n_points_x, self.n_points_t)
        dpinn_dt = dfdt(pinn, x, t, order=1).reshape(self.n_points_x, self.n_points_t)
        rhs = self._rhs(x, t, self.equation)

        x_times_n = torch.einsum("xt,n->xtn", x.reshape(self.n_points_x, self.n_points_t), self.n)
        test_x = torch.sin(math.pi*x_times_n)
        t_times_m = torch.einsum("xt,m->xtm", t.reshape(self.n_points_x, self.n_points_t), self.m)
        test_t = torch.sin(math.pi * t_times_m)
        test = torch.einsum("xtn,xtm->xtnm", test_x, test_t)
        test_x_dx = torch.pi * torch.einsum("n,xtn->xtn", self.n, torch.cos(torch.pi*x_times_n))
        test_dx = torch.einsum("xtn,xtm->xtnm", test_x_dx, test_t)
        test_t_dt = torch.pi * torch.einsum("m,xtm->xtm", self.m, torch.cos(torch.pi*t_times_m))
        test_dt = torch.einsum("xtn,xtm->xtnm", test_x, test_t_dt)

        loss1 = dx * dt * epsilon * torch.einsum("xt,xtnm->nm", dpinn_dx, test_dx)
        loss2 = dx * dt * epsilon * torch.einsum("xt,xtnm->nm", dpinn_dt, test_dt)
        loss3 = dx * dt * torch.einsum("xt,xtnm->nm", rhs, test)
        loss = loss1 + loss2 - loss3
        loss = loss**2 * self.G

        return loss.sum()
    
    def _rhs(self, x: torch.Tensor, t: torch.Tensor, equation: str):
        if equation == "sins":
            f1 = -4.0*torch.pi*torch.pi*torch.sin(2.0*torch.pi*x)*torch.sin(2.0*torch.pi*t)
            f2 = -4.0*torch.pi*torch.pi*torch.sin(2.0*torch.pi*x)*torch.sin(2.0*torch.pi*t)
            rhs = -f1-f2 # -Delta u = f so f = -Delta u, 0 on boundary the residual will be res = Delta u+f
            return rhs.reshape(self.n_points_x, self.n_points_t)
        if equation == "exp-sins":
            f1 = self.exact.dx2(x, t)
            f2 = self.exact.dt2(x, t)
            rhs = -f1-f2
            return rhs.reshape(self.n_points_x, self.n_points_t)
        raise ValueError(f"Invalid equation: {equation}")


class Error:
    # vm_norm is (v,v)_VM = epsilon (dv/dx,dvdx)+epsilon (dv/dy,dvdy)

    def __init__(self, epsilon: float, n_points_x: int, n_points_t: int, exact: ExactSolution, device: torch.device):
       self.epsilon = epsilon
       self.n_points_x = n_points_x
       self.n_points_t = n_points_t
       self.exact = exact
       self.x, self.t = middle_points((0,1), (0,1), n_points_x, n_points_t, True, device)
       self.l2_exact_norm = self._l2_exact_norm()
       self.vm_exact_norm = self._vm_exact_norm()

    def l2_norm(self, pinn: PINN) -> float:
        size = self.x.numel()
        diff = f(pinn, self.x, self.t)-self.exact(self.x, self.t)
        l2_z_norm = diff.pow(2).sum()/size
        l2_norm = math.sqrt(l2_z_norm) / self.l2_exact_norm
        return l2_norm
    
    def _l2_exact_norm(self) -> float:
        dx = 1.0 / self.n_points_x
        dt = 1.0 / self.n_points_t
        exact = self.exact(self.x, self.t)
        l2_exact_norm = exact.pow(2).sum()*dx*dt
        return math.sqrt(l2_exact_norm)

    def vm_norm(self, pinn: PINN) -> float:
        epsilon = self.epsilon
        x = self.x
        t = self.t
        dx = 1.0 / self.n_points_x
        dt = 1.0 / self.n_points_t
 
        dz_dx = dfdx(pinn, x, t, order=1).detach().flatten()
        dz_dt = dfdt(pinn, x, t, order=1).detach().flatten()
        x = x.detach().flatten()
        t = t.detach().flatten()

        exact_dx = self.exact.dx(x, t)
        diff_dx = dz_dx - exact_dx
        diff_dx_int = diff_dx.pow(2).sum()*dx*dt

        exact_dt = self.exact.dt(x, t)
        diff_dt = dz_dt - exact_dt
        diff_dt_int = diff_dt.pow(2).sum()*dx*dt

        vm_z_norm = epsilon*(diff_dx_int + diff_dt_int)

        vm_norm = math.sqrt(vm_z_norm) # / self.vm_exact_norm
        return vm_norm

    def _vm_exact_norm(self) -> float:
        dx = 1.0 / self.n_points_x
        dt = 1.0 / self.n_points_t
        x = self.x.detach().flatten()
        t = self.t.detach().flatten()
        exact_dx = self.exact.dx(x, t)
        exact_dx_norm = exact_dx.pow(2).sum()*dx*dt
        exact_dt = self.exact.dt(x, t)
        exact_dt_norm = exact_dt.pow(2).sum()*dx*dt
        vm_exact_norm = self.epsilon*(exact_dx_norm + exact_dt_norm).item()
        return math.sqrt(vm_exact_norm)
