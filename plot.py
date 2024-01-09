import argparse
import math
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import ticker
import torch
from src.exact import ExpSinsExactSolution, SinsExactSolution
import mpltools.annotation as mpl
from src.params import Params

from src.plot_utils import *

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rc('text', usetex=True)
font = {'family' : 'sans-serif', 'size' : 21}
matplotlib.rc('font', **font)

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
sqrt_loss_rel = np.sqrt(train_result.loss[pos_vec]) / train_result.vm_exact_norm
vm_norm_rel = train_result.vm_norm[pos_vec] / train_result.vm_exact_norm
##########################################################################

##########################################################################
# Loss and error
##########################################################################
fig, ax = plt.subplots()
fig.set_figwidth(7)
loss_label = r"$\frac{\sqrt{{\cal L \rm}_r^\phi(u_\theta)}}{\|u\|_U}$"
error_label = r"$\frac{\|u - u_\theta\|_U}{\|u\|_U}$"
ax.loglog(pos_vec, sqrt_loss_rel, '-',linewidth = 1.5, label=loss_label, color=loss_c)
ax.loglog(pos_vec, vm_norm_rel, '--', linewidth=1.5, label=error_label, color=error_c)
ax.legend(loc='lower left', labelcolor='linecolor')

# numticks = int(np.floor(np.log10(params.epochs)))
# locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,1.0, ))
# ax.xaxis.set_major_locator(locmaj)
# locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=numticks)
# ax.xaxis.set_minor_locator(locmin)
# ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" Error (estimates)")
save_fig(fig, "error-and-loss.png", tag)
save_fig(fig, "error-and-loss.pdf", tag)

##########################################################################
# Error to sqrt(loss)
##########################################################################
fig, ax = plt.subplots()
level = pos_vec[int(np.floor(len(pos_vec) * 0.2))]
ax.loglog(sqrt_loss_rel, vm_norm_rel, color=error_c, label="Error")
mpl.slope_marker((sqrt_loss_rel[level], 0.8*vm_norm_rel[level]), (1, 1), \
ax=ax, invert=False, poly_kwargs={'facecolor': 'white',
                                    'edgecolor':'black'})
# ax.loglog(np.sqrt(loss_vector[pos_vec]), np.sqrt(loss_vector[pos_vec]), color=loss_c, label="$y=x$")
ax.set_xlabel(r"Relative $\sqrt{Loss}$")
ax.set_ylabel(r"Relative Error ")
# ax.set_title(r"Error to $\sqrt{Loss}$")
save_fig(fig, "error-to-sqrt-loss.png", tag)
save_fig(fig, "error-to-sqrt-loss.pdf", tag)

##########################################################################
# H1 error
##########################################################################
fig, ax = plt.subplots()
ax.loglog(vm_norm_vector , '-',linewidth = 1.5, label=loss_label, color=loss_c)
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" H1 error")
save_fig(fig, "h1.png", tag)
save_fig(fig, "h1.pdf", tag)

##########################################################################
# L2 error
##########################################################################
fig, ax = plt.subplots()
ax.loglog(l2_norm_vector , '-',linewidth = 1.5, label=loss_label, color=loss_c)
ax.set_xlabel(r" Iterations ")
ax.set_ylabel(r" L2 error")
save_fig(fig, "l2.png", tag)
save_fig(fig, "l2.pdf", tag)

# Plot the solution in a "dense" mesh
n_x = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_y = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_x, n_y = torch.meshgrid(n_x, n_y, indexing="ij")
n_x_reshaped = n_x.reshape(-1, 1).requires_grad_(True)
n_y_reshaped = n_y.reshape(-1, 1).requires_grad_(True)

z = f(pinn, n_x_reshaped, n_y_reshaped).detach().reshape(PLOT_POINTS, PLOT_POINTS)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z)
ax.set_title("RVPINN solution")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.xaxis.set_ticks([0.0, 0.5, 1.0])
ax.yaxis.set_ticks([0.0, 0.5, 1.0])
colorbar = fig.colorbar(c, ax=ax)
colorbar.ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
save_fig(fig, "solution.png", tag)
# save_fig(fig, "solution.pdf", tag)

# Exact solution
z_exact = exact(n_x, n_y)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact)
ax.set_title("Exact solution")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.xaxis.set_ticks([0.0, 0.5, 1.0])
ax.yaxis.set_ticks([0.0, 0.5, 1.0])
colorbar = fig.colorbar(c, ax=ax)
colorbar.ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
save_fig(fig, "exact.png", tag)
# save_fig(fig, "exact.pdf", tag)

# Difference
z_exact = exact(n_x, n_y)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact - z)
ax.set_title("Exact - RVPINN")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.xaxis.set_ticks([0.0, 0.5, 1.0])
ax.yaxis.set_ticks([0.0, 0.5, 1.0])
colorbar = fig.colorbar(c, ax=ax)
colorbar.ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
save_fig(fig, "difference.png", tag)
# save_fig(fig, "difference.pdf", tag)

# Slice along t axis
n_t = torch.linspace(0.0, 1.0, steps=PLOT_POINTS)
n_x = torch.full_like(n_t, 0.25)
z_pinn = f(pinn, n_x.reshape(-1, 1), n_t.reshape(-1, 1)).detach().reshape(-1)
z_exact = exact(n_x, n_t)
fig, ax = plt.subplots()
ax.set_title("Slice along t axis at x=0.25")
ax.plot(n_t, z_pinn, "-", label="RVPINN", color=vpinn_c, linewidth = 2)
ax.plot(n_t, z_exact, "--", label="Analytical", color=analytical_c, linewidth=2)
ax.legend(labelcolor='linecolor')
ax.set_xlabel("$t$")
ax.set_ylabel("$u$")
ax.xaxis.set_ticks([0.0, 0.5, 1.0])
save_fig(fig, "t-slice.png", tag)
save_fig(fig, "t-slice.pdf", tag)

plt.show()
