import math
import torch


def exact_solution(x, t, epsilon) -> torch.Tensor:
  r1 = (1.0 + math.sqrt(1.0 + 4.0*epsilon*epsilon*math.pi*math.pi))/(2.0*epsilon)
  r2 = (1.0 - math.sqrt(1.0 + 4.0*epsilon*epsilon*math.pi*math.pi))/(2.0*epsilon)
  res_t = (torch.exp(r1*(t-1.0))-torch.exp(r2*(t-1.0)))/(math.exp(-r1)-math.exp(-r2))
  res_x = torch.sin(math.pi*x)
  res = res_t.mul(res_x)
  return res

def exact_solution_dx(x, t, epsilon) -> torch.Tensor:
  r1 = (1.0 + math.sqrt(1.0 + 4.0*epsilon*epsilon*math.pi*math.pi))/(2.0*epsilon)
  r2 = (1.0 - math.sqrt(1.0 + 4.0*epsilon*epsilon*math.pi*math.pi))/(2.0*epsilon)
  res_t = (torch.exp(r1*(t-1.0))-torch.exp(r2*(t-1.0)))/(math.exp(-r1)-math.exp(-r2))
  res_x_dx = math.pi*torch.cos(math.pi*x)
  res = res_t.mul(res_x_dx)
  return res

def exact_solution_dt(x, t, epsilon) -> torch.Tensor:
  r1 = (1.0 + math.sqrt(1.0 + 4.0*epsilon*epsilon*math.pi*math.pi))/(2.0*epsilon)
  r2 = (1.0 - math.sqrt(1.0 + 4.0*epsilon*epsilon*math.pi*math.pi))/(2.0*epsilon)
  res_t_dt = (r1*torch.exp(r1*(t-1.0))-r2*torch.exp(r2*(t-1.0)))/(math.exp(-r1)-math.exp(-r2))
  res_x = torch.sin(math.pi*x)
  res = res_t_dt.mul(res_x)
  return res