from typing import Tuple
import torch


def middle_points(x_domain: Tuple[int, int],
                  t_domain: Tuple[int, int], 
                  n_points_x: int,
                  n_points_t: int,
                  requires_grad: bool, 
                  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    h_x = (x_domain[1] - x_domain[0]) / (2 * n_points_x)
    x_raw = torch.linspace(x_domain[0] + h_x, x_domain[1] - h_x, n_points_x)
    h_t = (t_domain[1] - t_domain[0]) / (2 * n_points_t)
    t_raw = torch.linspace(t_domain[0] + h_t, t_domain[1] - h_t, n_points_t)
    x, t = torch.meshgrid(x_raw, t_raw, indexing="ij")
    x = x.reshape(-1, 1).to(device).requires_grad_(requires_grad)
    t = t.reshape(-1, 1).to(device).requires_grad_(requires_grad)
    return x, t
