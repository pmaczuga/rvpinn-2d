from abc import ABC
from typing import Any
import torch
from torch import Tensor, pi, sin, cos, exp

class ExactSolution(ABC):
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def dx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class SinsExactSolution(ExactSolution):
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sins = torch.sin(2*torch.pi*x) * torch.sin(2.0*torch.pi*t)
        res = sins
        return res

    def dx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sins = 2*torch.pi*torch.cos(2*torch.pi*x) * torch.sin(2.0*torch.pi*t)
        res = sins
        return res
    
    def dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sins = 2*torch.pi*torch.sin(2*torch.pi*x) * torch.cos(2.0*torch.pi*t)
        res = sins
        return res
    

class ExpSinsExactSolution(ExactSolution):
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = -exp(pi*(x - 2*t))
        sins = sin(2*torch.pi*x) * sin(pi*t)
        res = exp1 * sins
        return res
    
    def dx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = -pi * exp(pi*(x-2*t)) * sin(pi*t)
        sins = sin(2*pi*x) + 2*cos(2*pi*x)
        res = exp1 * sins
        return res

    def dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = -exp(pi*(x-2*t)) * sin(2*pi*x)
        sins = cos(pi*t) - 2*sin(pi*t)
        res = exp1 * sins
        return res
    
    def dx2(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = -pi*pi*exp(pi*(x-2*t)) * sin(pi*t)
        sins = 4*cos(2*pi*x) - 3*sin(2*pi*x)
        res = exp1 * sins
        return res
    
    def dt2(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = pi*pi*exp(pi*(x-2*t)) * sin(2*pi*x)
        sins = 4*cos(pi*t) - 3*sin(pi*t)
        res = exp1 * sins
        return res
