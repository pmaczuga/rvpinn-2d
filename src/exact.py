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
        exp1 = 4.0*torch.pi*torch.pi * torch.exp(torch.pi*(x-2*t)) * torch.sin(torch.pi * t)
        sin1 = torch.sin(2*torch.pi*x)
        res1 = exp1 * sin1
        exp2 = 4.0*torch.pi*torch.pi* torch.exp(torch.pi*(x-2*t)) * torch.sin(torch.pi * t)
        sin2 = torch.cos(2*torch.pi*x)
        res2 = exp2 * sin2
        return res1 + res2
    
    def dt2(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = 4.0*torch.pi*torch.pi * torch.exp(torch.pi*(x-2*t)) *torch.sin(torch.pi * t)
        sin1 = torch.cos(2*torch.pi*x)
        res1 = exp1 * sin1
        exp2 = 4.0*torch.pi*torch.pi * torch.exp(torch.pi*(x-2*t)) * torch.cos(torch.pi * t)
        sin2 = torch.sin(2*torch.pi*x)
        res2 = exp2 * sin2
        return res1 + res2
    