import torch


def exact_solution(x, y, epsilon) -> torch.Tensor:
    exp = -torch.exp(torch.pi*(x - 2*y))
    sins = torch.sin(2*torch.pi*x) * torch.sin(torch.pi*y)
    res = exp * sins
    return res

def exact_solution_dx(x, y, epsilon) -> torch.Tensor:
    exp = -torch.pi * torch.exp(torch.pi*(x-2*y)) * torch.sin(torch.pi * y)
    sins = torch.sin(2*torch.pi*x) + 2*torch.cos(2*torch.pi*x)
    res = exp * sins
    return res

def exact_solution_dy(x, y, epsilon) -> torch.Tensor:
    exp = -torch.pi * torch.exp(torch.pi*(x-2*y)) * torch.sin(torch.pi * y)
    sins = torch.sin(2*torch.pi*x) + 2*torch.cos(2*torch.pi*x)
    res = exp * sins
    return res
