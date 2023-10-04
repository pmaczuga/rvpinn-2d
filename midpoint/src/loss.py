import math
import torch
from src.pinn_core import *
from src.exact import *

def shift(x, t) -> torch.Tensor:
  shift_x = torch.sin(math.pi*x)
  shift_t =(1.0-t)
  res = shift_x.mul(shift_t)
  return res

def shift_dx(x, t) -> torch.Tensor:
  shift_x_dx = math.pi*torch.cos(math.pi*x)
  shift_t =(1.0-t)
  res = shift_x_dx.mul(shift_t)
  return res

def shift_dt(x, t) -> torch.Tensor:
  shift_x = torch.sin(math.pi*x)
  shift_t_dt =-1.0
  res = shift_x *shift_t_dt
  return res

def interior_loss(pinn: PINN, x:torch.Tensor, t: torch.tensor, epsilon: float):
    device = x.device
    dx = 1.0 / len(x)
    dt = 1.0 / len(t)

    # norm is (v,v)_VM = epsilon (dv/dx,dvdx)+epsilon (dv/dy,dvdy)

    final_loss = 0.0
    for n in range(1,20):
      test_x = torch.sin(n*math.pi*x)
      test_x_dx = n * math.pi * torch.cos(n*math.pi*x)
      constant_1b = (2.0*math.pi*n-math.sin(2.0*math.pi*n))/(4.0*math.pi*n)
      constant_2a = epsilon * n*n*math.pi*math.pi *(2.0*math.pi*n+math.sin(2.0*math.pi*n))/(4.0*math.pi*n)
      for m in range(1,20):
        test_t = torch.sin(m*math.pi*t)
        test_t_dt = m * math.pi * torch.cos(m*math.pi*t)
        test_dt = test_x.mul(test_t_dt)
        test_dx = test_t.mul(test_x_dx)
        test = test_x.mul(test_t)
        constant_1a = epsilon * m*m*math.pi*math.pi *(2.0*math.pi*m+math.sin(2.0*math.pi*m))/(4.0*math.pi*m)
        constant_2b = (2.0*math.pi*m-math.sin(2.0*math.pi*m))/(4.0*math.pi*m)
        constant_full = constant_1a*constant_1b + constant_2a*constant_2b
        constant = 1.0 / constant_full
        u0_dx = math.pi * torch.cos(math.pi * x )
        u0_dx_t = u0_dx.mul(1 - t)
        u0_x_dt = (-1.0) * torch.sin (math.pi * x)
        loss_tmp_weak = \
          + epsilon * dfdt(pinn, x.to(device), t.to(device), order=1).to(device) * test_dt.to(device) * dx * dt \
          + epsilon * dfdx(pinn, x.to(device), t.to(device), order=1).to(device) * test_dx.to(device) *  dx * dt \
          + dfdt(pinn, x, t).to(device) * test.to(device) * dx * dt \
          + epsilon * u0_dx_t.to(device) * test_dx.to(device)  * dx * dt \
          + epsilon * u0_x_dt.to(device) * test_dt.to(device)  * dx * dt \
          + u0_x_dt.to(device) * test.to(device) * dx * dt
        final_loss+=  \
          loss_tmp_weak.sum().pow(2) * constant

    loss = final_loss
    return loss


def compute_loss(
    pinn: PINN, epsilon: float, x: torch.Tensor = None, t: torch.Tensor = None,
    x_init: torch.Tensor = None,
    weight_f = 1.0, weight_b = 1.0, weight_i = 1.0,
    verbose = False, length=1.0, total_time=1.0
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """
    device = x.device
    final_loss = \
        weight_f * interior_loss(pinn, x, t, epsilon)

#PRINTING
    number = 40
    number2 = 1600

    delta = 0.5*1.0/(40)
    x_domain_loc = [0.0, length]
    t_domain_loc = [0.0, total_time]
    x_raw_loc = torch.linspace(x_domain_loc[0]+delta, x_domain_loc[1]-delta, steps=40, requires_grad=True)
    t_raw_loc = torch.linspace(t_domain_loc[0]+delta, t_domain_loc[1]-delta, steps=40, requires_grad=True)
    grids_loc = torch.meshgrid(x_raw_loc, t_raw_loc, indexing="ij")
    x_loc = grids_loc[0].flatten().reshape(-1, 1).to(device)
    t_loc = grids_loc[1].flatten().reshape(-1, 1).to(device)
    z_loc = f(pinn.to(device), x_loc, t_loc)+shift(x_loc,t_loc)#-exact_solution(x_loc,t_loc)

    #dzdx
    dz_dx = dfdx(pinn.to(device), x_loc, t_loc, order=1)
    #dfdx jest duze w pierwszym i ostatnim punkcie number2-1
    for i in range(0,number2):
      if dz_dx[i]>1:
        print(dz_dx[i])
        print(i)
      if dz_dx[i]<-1:
        print(dz_dx[i])
        print(i)
    dz_dx0 =dz_dx[0]
    dz_dx9999 =dz_dx[number2-1]
    dz_dx_norm = ((dz_dx.pow(2).sum()-dz_dx0*dz_dx0-dz_dx9999*dz_dx9999)/number2)
#    print(f'dz_dx_norm:{dz_dx_norm.item():.5f}')

    #dzdt
    dz_dt = dfdt(pinn.to(device), x_loc, t_loc, order=1)
    #dfdt jest duze
    for i in range(0,number2):
      if dz_dt[i]>1:
        print(dz_dt[i])
        print(i)
      if dz_dt[i]<-1:
        print(dz_dt[i])
        print(i)
    dz_dt0 =dz_dt[0]
    dz_dt9999 =dz_dt[number2-1]
    dz_dt_norm = ((dz_dt.pow(2).sum()-dz_dt9999*dz_dt9999-dz_dt0*dz_dt0)/number2)
#    print(f'dz_dt_norm:{dz_dt_norm.item():.5f}')

    #exact_dx
    exact_dx = exact_solution_dx(x_loc,t_loc, epsilon)
    exact_dx_norm = (exact_dx.pow(2).sum()/number2)
#    print(f'exact_dx_norm:{exact_dx_norm:.5f}')

    #exact_dt
    exact_dt = exact_solution_dt(x_loc,t_loc, epsilon)
    exact_dt_norm = (exact_dt.pow(2).sum()/number2)
#    print(f'exact_dt_norm:{exact_dt_norm:.5f}')

    #shift_dx
    shift_dx_all = shift_dx(x_loc,t_loc)
    shift_dx_norm = (shift_dx_all.pow(2).sum()/number2)
#    print(f'shift_dx_norm:{shift_dx_norm:.5f}')

    #shift_dt
    shift_dt_all = shift_dt(x_loc,t_loc)
    shift_dt_norm = (shift_dt_all.pow(2).sum()/number2)
#    print(f'shift_dt_norm:{shift_dt_norm:.5f}')

    VM_z_norm = epsilon*math.sqrt((dz_dx_norm+dz_dt_norm-exact_dx_norm-exact_dt_norm+shift_dx_norm+shift_dt_norm)**2)
    VM_exact_norm = epsilon*math.sqrt((exact_dx_norm+exact_dt_norm)**2)

#    print(f'VM_z_norm:{VM_z_norm:.5f}')
#    print(f'VM_exact_norm:{VM_exact_norm:.5f}')
    VM_norm = VM_z_norm / VM_exact_norm
#    print(f'VM_norm:{VM_norm:.5f}')

    z = f(pinn.to(device), x_loc, t_loc)+shift(x_loc,t_loc)-exact_solution(x_loc,t_loc, epsilon)
    exact = exact_solution(x_loc,t_loc, epsilon) #the values are (0,1)
    L2_exact_norm = exact.pow(2).sum()/number2 #the values are (0,1) we average them
    L2_z_norm = z.pow(2).sum()/number2
    L2_norm = math.sqrt(L2_z_norm)/math.sqrt(L2_exact_norm)
#    print(f'L2_z_norm:{L2_z_norm:.5f}')
#    print(f'L2_exact_norm:{L2_exact_norm:.5f}')
#    print(f'L2_z_norm/l2_norm:{L2_z_norm/L2_norm:.5f}')

    #loss
    interior_loss_val = interior_loss(pinn, x, t, epsilon)

    with open('OutputPlots.txt', 'a') as file_f:
      print(f"||u_exact||VM: {VM_exact_norm:.7f} ||u_NN-u_exact||VM: {VM_z_norm:.7f} ||u_NN-u_exact||VM/||u_exact||VM: {VM_norm:.7f} ||u_exact||L2: {L2_exact_norm:.7f} ||u_NN-u_exact||L2: {L2_z_norm:.7f} ||u_NN-u_exact||L2/||u_exact||L2: {L2_norm:.7f} sqrt(InteriorLoss)/||u_exact||VM: {float(math.sqrt(interior_loss_val)/VM_exact_norm):.7f} sqrt(InteriorLoss)/||u_exact||L2: {float(math.sqrt(interior_loss_val)/L2_exact_norm):.7f} sqrt(InteriorLoss): {float(math.sqrt(interior_loss_val)):.7f} InteriorLoss: {float(interior_loss_val):.7f} ", file=file_f)

    if not verbose:
        return final_loss
    else:
        return final_loss, interior_loss(pinn, x, t, epsilon), initial_loss(pinn, x_init, t), finito_loss(pinn, x_init, t), bottom_loss(pinn, x_init, t), top_loss(pinn, x_init, t)