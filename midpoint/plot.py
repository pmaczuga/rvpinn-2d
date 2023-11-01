import math
from matplotlib import pyplot as plt
import torch
from src.exact import exact_solution
from src.loss import shift

from src.plot_utils import *
from params import *


pinn = torch.load("results/data/pinn.pt")
loss_values = torch.load("results/data/loss_values.pt")
exec_time = torch.load("results/data/exec_time.pt")
train_result: TrainResult = torch.load("results/data/train_result.pt")
x_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]
x_init_raw = torch.linspace(0.0, 1.0, steps=N_POINTS_INIT)

def decreasing_values(y):
    decreasing=y
    for i in range(2, len(y)) :
      decreasing[i]=min(y[i],decreasing[i-1])
    return decreasing

average_loss = running_average(loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (runnig average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss)
ax.set_yscale('log')
fig.savefig("results/loss.png")
fig.savefig("results/loss.pdf")

x_raw = torch.linspace(x_domain[0], x_domain[1], steps=1000, requires_grad=True)
t_raw = torch.linspace(t_domain[0], t_domain[1], steps=1000, requires_grad=True)
grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
x = grids[0].flatten().reshape(-1, 1)
t = grids[1].flatten().reshape(-1, 1)
z = f(pinn, x, t)
color = plot_color(z, x, t, 1000, 1000)
color.savefig("results/rvpinn.png")
color.savefig("results/rvpinn.pdf")

x_raw = torch.linspace(x_domain[0], x_domain[1], steps=1000, requires_grad=True)
t_raw = torch.linspace(t_domain[0], t_domain[1], steps=1000, requires_grad=True)
grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
x = grids[0].flatten().reshape(-1, 1)
t = grids[1].flatten().reshape(-1, 1)
z = exact_solution(x,t, EPSILON)
color = plot_color(z, x, t, 1000, 1000)
color.savefig("results/exact.png")
color.savefig("results/exact.pdf")

x_init = torch.linspace(0.0, 1.0, steps=1000)
x_init = x_init*LENGTH
pinn_init = f(pinn, torch.zeros_like(x_init_raw).reshape(-1,1)+0.5, x_init_raw.reshape(-1, 1)) #x=0.5, t between (-1,1)
fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
ax.set_title("Solution profile at x=0.5")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.plot(x_init_raw, pinn_init.flatten().detach(), label="PINN solution")
ax.legend()
fig.savefig("results/profile.png")
fig.savefig("results/profile.pdf")

from IPython.display import HTML
ani = plot_solution(pinn, x, t)
ani.save("results/animation.mp4")

x_raw = torch.linspace(x_domain[0], x_domain[1], steps=100, requires_grad=True)
t_raw = torch.linspace(t_domain[0], t_domain[1], steps=100, requires_grad=True)
grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
x = grids[0].flatten().reshape(-1, 1)
t = grids[1].flatten().reshape(-1, 1)
z = f(pinn, x, t)+shift(x,t)-exact_solution(x,t, EPSILON)
exact = exact_solution(x,t, EPSILON) #the values are (0,1)
exact_norm = exact.pow(2).sum()/10000 #the values are (0,1) we average them
z_norm = z.pow(2).sum()/10000
z1 = EPSILON*torch.sqrt(z/exact_norm)
l2_norm = math.sqrt(z_norm)/math.sqrt(exact_norm)
print(f'l2_z_norm:{z_norm:.5f}')
print(f'l2_exact_norm:{exact_norm:.5f}')
print(f'z_norm/l2_norm:{z_norm/l2_norm:.5f}')
color = plot_color(z1.cpu(), x.cpu(), t.cpu(), 100, 100)
color.savefig("results/error.png")
color.savefig("results/error.pdf")

decreasing_loss = decreasing_values(loss_values)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(decreasing_loss)
ax.set_yscale('log')
fig.savefig("results/loss.png")
fig.savefig("results/loss.pdf")

plt.show()
