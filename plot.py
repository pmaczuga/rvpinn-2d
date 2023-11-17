import argparse
import math
from matplotlib import pyplot as plt
import matplotlib
import torch
from src.exact import ExpSinsExactSolution, SinsExactSolution
import mpltools.annotation as mpl
from src.params import Params

from src.plot_utils import *

# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

parser = argparse.ArgumentParser(
                    prog='RVPINN',
                    description='Runs the training of RVPINN')
parser.add_argument('--tag', type=str)
args = parser.parse_args()
tag = args.tag if args.tag is not None else Params().tag
params = Params(f"results/{tag}/params.ini")

pinn = torch.load(f"results/{tag}/data/pinn.pt", map_location=torch.device("cpu"))
exec_time = torch.load(f"results/{tag}/data/exec_time.pt", map_location=torch.device("cpu"))
train_result: TrainResult = torch.load(f"results/{tag}/data/train_result.pt", map_location=torch.device("cpu"))
x_domain = [0.0, 1.0]
t_domain = [0.0, 1.0]
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

PLOT_POINTS = 1000

if (params.equation == "sins"):
  exact = SinsExactSolution()
elif (params.equation == "exp-sins"):
  exact = ExpSinsExactSolution()

##########################################################################
vec = train_result.loss
best = math.inf
best_vec = [1.]
pos_vec = [1.]
epochs_vector = np.array(range(1, params.epochs + 1))


for n in range(params.epochs):
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
ax.loglog(pos_vec, vm_norm_vector[pos_vec] / train_result.vm_exact_norm, '--', linewidth=1.5, label=error_label, color=error_c)
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
ax.set_title("PINN solution, eps={}".format(params.epsilon))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/solution")

# Exact solution
z_exact = exact(n_x, n_y)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact)
ax.set_title("Exact solution, eps={}".format(params.epsilon))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/exact")

# Difference
z_exact = exact(n_x, n_y)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact - z)
ax.set_title("Exact - PINN, eps={}".format(params.epsilon))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/difference")


# Slice along t axis
n_t = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_x = torch.full_like(n_t, 0.25)
z_pinn = f(pinn, n_x.reshape(-1, 1), n_t.reshape(-1, 1)).detach().reshape(-1)
z_exact = exact(n_x, n_t)
fig, ax = plt.subplots()
ax.set_title("Slice along t axis at x=0.25, eps={}".format(params.epsilon))
ax.plot(n_t, z_pinn, "--", label="PINN")
ax.plot(n_t, z_exact, label="Exact")
ax.legend()
ax.set_xlabel("t")
fig.savefig(f"results/x_slice")

##########################################################################
# Error to sqrt(loss)
##########################################################################
fig, ax = plt.subplots()
level = pos_vec[int(np.floor(len(pos_vec) * 0.2))]
ax.loglog(np.sqrt(loss_vector[pos_vec]) / train_result.vm_exact_norm, train_result.vm_norm[pos_vec], color=error_c, label="Error")
mpl.slope_marker((loss_vector[level]**(1/2) / train_result.vm_exact_norm, 0.8*train_result.vm_norm[level]), (1, 1), \
ax=ax, invert=False, poly_kwargs={'facecolor': 'white',
                                    'edgecolor':'black'})
# ax.loglog(np.sqrt(loss_vector[pos_vec]), np.sqrt(loss_vector[pos_vec]), color=loss_c, label="$y=x$")
ax.set_xlabel(r"Relative $\sqrt{Loss}$")
ax.set_ylabel(r"Relative Error ")
# ax.set_title(r"Error to $\sqrt{Loss}$")
fig.savefig("results/error-to-sqrt-loss.png")
fig.savefig("results/error-to-sqrt-loss.pdf")

plt.show()
