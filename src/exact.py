import torch


def exact_solution(x, y, epsilon) -> torch.Tensor:
  sins = torch.sin(2*torch.pi*x) * torch.sin(2.0*torch.pi*y)
  res = sins
  return res

def exact_solution_dx(x, y, epsilon) -> torch.Tensor:
  sins = 2*torch.pi*torch.cos(2*torch.pi*x) * torch.sin(2.0*torch.pi*y)
  res = sins
  return res

def exact_solution_dy(x, y, epsilon) -> torch.Tensor:
  sins = 2*torch.pi*torch.sin(2*torch.pi*x) * torch.cos(2.0*torch.pi*y)
  res = sins
  return res
