from abc import ABC
from typing import Any
import torch

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
        exp = -torch.exp(torch.pi*(x - 2*t))
        sins = torch.sin(2*torch.pi*x) * torch.sin(torch.pi*t)
        res = exp * sins
        return res
    
    def dx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = -2.0*torch.pi * torch.exp(torch.pi*(x-2*t)) * torch.sin(torch.pi * t)
        sin1 = torch.sin(2*torch.pi*x)
        res1 = exp1 * sin1
        exp2 = -2.0*torch.pi * torch.exp(torch.pi*(x-2*t)) * torch.sin(torch.pi * t)
        sin2 = torch.cos(2*torch.pi*x)
        res2 = exp2 * sin2
        return res1 + res2

    def dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        exp1 = -torch.pi * torch.exp(torch.pi*(x-2*t)) * torch.sin(torch.pi* t)
        sin1 = torch.cos(2*torch.pi*x)
        res1 = exp1 * sin1
        exp2 = -torch.pi * torch.exp(torch.pi*(x-2*t)) * torch.cos(torch.pi * t)
        sin2 = torch.sin(2*torch.pi*x)
        res2 = exp2 * sin2
        return res1 + res2
    
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
    