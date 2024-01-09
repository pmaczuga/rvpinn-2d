from typing import Tuple
import torch

def linearized(ix, it, n):
  return ix * n + it

def calc_A(n, device) -> torch.Tensor:
    m = n+1
    h = 1.0 / m
    A = torch.zeros(n*n, m*m)
    for ix in range(n):
        for it in range(n):
            i = linearized(ix, it, n)
            for jx in range(m):
                for jt in range(m):
                    j = linearized(jx, jt, m)
                    if (jx == ix or jx == ix+1) and (jt == it or jt == it+1):
                        A[i, j] = h**2 * 0.25
    return A.to(device)

def calc_S(n, device) -> Tuple[torch.Tensor, torch.Tensor]:
    m = n+1
    h = 1.0 / m
    Sx = torch.zeros(n*n, m*m)
    St = torch.zeros(n*n, m*m)
    for ix in range(n):
        for it in range(n):
            i = linearized(ix, it, n)
            for jx in range(m):
                for jt in range(m):
                    j = linearized(jx, jt, m)
                    if (jx == ix and jt == it):
                        Sx[i, j] =  h / 2.0
                        St[i, j] =  h / 2.0
                    if (jx == ix+1 and jt == it):
                        Sx[i, j] = -h / 2.0
                        St[i, j] =  h / 2.0
                    if (jx == ix and jt == it+1):
                        Sx[i, j] =  h / 2.0
                        St[i, j] = -h / 2.0
                    if (jx == ix+1 and jt == it+1):
                        Sx[i, j] = -h / 2.0
                        St[i, j] = -h / 2.0
    return Sx.to(device), St.to(device)

def calc_G(n, device) -> torch.Tensor:
    G = torch.zeros(n*n, n*n)
    for ix in range(n):
        for it in range(n):
            i = linearized(ix, it, n)
            for jx in range(n):
                for jt in range(n):
                    j = linearized(jx, jt, n)
                    if (ix == jx and it == jt):
                        G[i, j] = 8/3                        
                    if (ix == jx and abs(it-jt) == 1) or (it == jt and abs(ix-jx) == 1):
                        G[i, j] = -1/3
                    if abs(ix-jx) == 1 and abs(it-jt) == 1:
                        G[i, j] = -1/3
    return G.to(device)

def calc_G_T(n, device):
    return calc_G(n, device).inverse()