import math
from matplotlib import pyplot as plt
import torch
from src.exact import exact_solution

from src.plot_utils import *
from params import *


pinn = torch.load("results/data/pinn.pt")
loss_values = torch.load("results/data/loss_values.pt")
exec_time = torch.load("results/data/exec_time.pt")
train_result: TrainResult = torch.load("results/data/train_result.pt")
x_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]
x_init_raw = torch.linspace(0.0, 1.0, steps=1000)
loss_vector = train_result.loss
vm_norm_vector = train_result.vm_norm
l2_norm_vector = train_result.l2_norm

# Colors
vpinn_c = "#0B33B5"
analytical_c = "#D00000"
loss_c = "darkorange"
norm_c = "#58106a"
error_c = "darkgreen"

##########################################################################
vec = train_result.loss
best = math.inf
best_vec = [1.]
pos_vec = [1.]
epochs_vector = np.array(range(1, EPOCHS + 1))


for n in range(EPOCHS):
  if vec[n]<best and vec[n]>0:
    best_vec.append(vec[n])
    pos_vec.append(n+1)
    best = 1*vec[n]

pos_vec = np.array(pos_vec, dtype=int) - 1
##########################################################################

##########################################################################
# Loss and error
##########################################################################
fig, ax = plt.subplots()
loss_label = r"$\frac{\sqrt{{\cal L \rm}_r^\phi(u_\theta)}}{\|u\|_U}$"
error_label = r"$\frac{\|u - u_\theta\|_U}{\|u\|_U}$"
ax.loglog(pos_vec, np.sqrt(loss_vector[pos_vec]) / train_result.vm_exact_norm, '-',linewidth = 1.5, label=loss_label, color=loss_c)
ax.loglog(pos_vec, vm_norm_vector[pos_vec], '--', linewidth=1.5, label=error_label, color=error_c)
ax.legend(loc='lower left', labelcolor='linecolor')
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Error (estimates)")
fig.savefig("results/error-and-loss.png")
fig.savefig("results/error-and-loss.pdf")

##########################################################################
# H1 error
##########################################################################
fig, ax = plt.subplots()
ax.plot(vm_norm_vector , '-',linewidth = 1.5, label=loss_label, color=loss_c)
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" H1 error")
fig.savefig("results/h1.png")
fig.savefig("results/h1.pdf")

##########################################################################
# L2 error
##########################################################################
fig, ax = plt.subplots()
ax.plot(vm_norm_vector , '-',linewidth = 1.5, label=loss_label, color=loss_c)
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" L2 error")
fig.savefig("results/l2.png")
fig.savefig("results/l2.pdf")

# Plot the solution in a "dense" mesh
n_x = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_y = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_x, n_y = torch.meshgrid(n_x, n_y)
n_x_reshaped = n_x.reshape(-1, 1).requires_grad_(True)
n_y_reshaped = n_y.reshape(-1, 1).requires_grad_(True)

z = f(pinn, n_x_reshaped, n_y_reshaped).detach().reshape(PLOT_POINTS, PLOT_POINTS)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z)
ax.set_title("PINN solution, eps={}".format(EPSILON))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/solution")

# Exact solution
z_exact = exact_solution(n_x, n_y, EPSILON)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact)
ax.set_title("Exact solution, eps={}".format(EPSILON))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/exact")

# Difference
z_exact = exact_solution(n_x, n_y, EPSILON)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact - z)
ax.set_title("Exact - PINN, eps={}".format(EPSILON))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/difference")

# Initial solution
n_x = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_t = torch.zeros_like(n_x)
z_pinn = f(pinn, n_x.reshape(-1, 1), n_t.reshape(-1, 1)).detach().reshape(-1)
z_exact = torch.sin(n_x * math.pi)
fig, ax = plt.subplots()
ax.set_title("Initial condition")
ax.plot(n_x, z_pinn, "-", label="PINN")
ax.plot(n_x, z_exact, "--", label="Exact")
ax.legend()
ax.set_xlabel("x")
fig.savefig(f"results/initial")

# Slice along t axis
n_t = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_x = torch.full_like(n_t, 0.5)
z_pinn = f(pinn, n_x.reshape(-1, 1), n_t.reshape(-1, 1)).detach().reshape(-1)
z_exact = exact_solution(n_x, n_t, EPSILON)
fig, ax = plt.subplots()
ax.set_title("Slice along t axis at x=0.5, eps={}".format(EPSILON))
ax.plot(n_t, z_pinn, "--", label="PINN")
ax.plot(n_t, z_exact, label="Exact")
ax.legend()
ax.set_xlabel("t")
fig.savefig(f"results/x_slice")

plt.show()