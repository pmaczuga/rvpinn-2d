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
    dx = 1.0 / len(x)
    dt = 1.0 / len(t)
    device = x.device

    # norm is (v,v)_VM = epsilon (dv/dx,dvdx)+epsilon (dv/dy,dvdy)

    final_loss = 0.0
    #(x,t) is the center of element
    #(x,t) -> (x1,t1)=(x+0.5*h,t+0.5*t)
    #(x,t) -> (x2,t2)=(x-0.5*h,t+0.5*t)
    #(x,t) -> (x3,t3)=(x+0.5*h,t-0.5*t)
    #(x,t) -> (x4,t4)=(x-0.5*h,t-0.5*t)
    #values * 0.5
    hx = dx
    ht = dt
    # https://www.minhnguyencae.info/2d-gaussian-quadrature/
    #(3^(1/3)/3)*0.5*h (since we map [-1,1] into [0,h])
    # weights = 1
    #point = 0.23112042478354490161445075826199
    point = (math.sqrt(3.0)/3.0)*0.5
    x1 = x+point*hx
    x2 = x-point*hx
    x3 = x+point*hx
    x4 = x-point*hx
    t1 = t+point*ht
    t2 = t-point*ht
    t3 = t+point*ht
    t4 = t-point*ht

    for n in range(1,20):
      test_x1 = torch.sin(n*math.pi*x1)
      test_x1_dx = n * math.pi * torch.cos(n*math.pi*x1)
      test_x2 = torch.sin(n*math.pi*x2)
      test_x2_dx = n * math.pi * torch.cos(n*math.pi*x2)
      test_x3 = torch.sin(n*math.pi*x3)
      test_x3_dx = n * math.pi * torch.cos(n*math.pi*x3)
      test_x4 = torch.sin(n*math.pi*x4)
      test_x4_dx = n * math.pi * torch.cos(n*math.pi*x4)
      constant_1b = (2.0*math.pi*n-math.sin(2.0*math.pi*n))/(4.0*math.pi*n)
      constant_2a = epsilon * n*n*math.pi*math.pi *(2.0*math.pi*n+math.sin(2.0*math.pi*n))/(4.0*math.pi*n)
      for m in range(1,20):
        test_t1 = torch.sin(m*math.pi*t1)
        test_t1_dt = m * math.pi * torch.cos(m*math.pi*t1)
        test_t2 = torch.sin(m*math.pi*t2)
        test_t2_dt = m * math.pi * torch.cos(m*math.pi*t2)
        test_t3 = torch.sin(m*math.pi*t3)
        test_t3_dt = m * math.pi * torch.cos(m*math.pi*t3)
        test_t4 = torch.sin(m*math.pi*t4)
        test_t4_dt = m * math.pi * torch.cos(m*math.pi*t4)
        test1_dt = test_x1.mul(test_t1_dt)
        test1_dx = test_t1.mul(test_x1_dx)
        test2_dt = test_x2.mul(test_t2_dt)
        test2_dx = test_t2.mul(test_x2_dx)
        test3_dt = test_x3.mul(test_t3_dt)
        test3_dx = test_t3.mul(test_x3_dx)
        test4_dt = test_x4.mul(test_t4_dt)
        test4_dx = test_t4.mul(test_x4_dx)
        test1 = test_x1.mul(test_t1)
        test2 = test_x2.mul(test_t2)
        test3 = test_x3.mul(test_t3)
        test4 = test_x4.mul(test_t4)
        constant_1a = epsilon * m*m*math.pi*math.pi *(2.0*math.pi*m+math.sin(2.0*math.pi*m))/(4.0*math.pi*m)
        constant_2b = (2.0*math.pi*m-math.sin(2.0*math.pi*m))/(4.0*math.pi*m)
        constant_full = constant_1a*constant_1b + constant_2a*constant_2b
        constant = 1.0 / constant_full
        u0_dx1 = math.pi * torch.cos(math.pi * x1 )
        u0_dx1_t1 = u0_dx1.mul(1 - t1)
        u0_x1_dt1 = (-1.0) * torch.sin (math.pi * x1)
        u0_dx2 = math.pi * torch.cos(math.pi * x2 )
        u0_dx2_t2 = u0_dx2.mul(1 - t2)
        u0_x2_dt2 = (-1.0) * torch.sin (math.pi * x2)
        u0_dx3 = math.pi * torch.cos(math.pi * x3 )
        u0_dx3_t3 = u0_dx3.mul(1 - t3)
        u0_x3_dt3 = (-1.0) * torch.sin (math.pi * x3)
        u0_dx4 = math.pi * torch.cos(math.pi * x4 )
        u0_dx4_t4 = u0_dx4.mul(1 - t4)
        u0_x4_dt4 = (-1.0) * torch.sin (math.pi * x4)
        loss_tmp_weak1 = \
          + epsilon * dfdt(pinn, x1.to(device), t1.to(device), order=1).to(device) * test1_dt.to(device) * dx * dt \
          + epsilon * dfdx(pinn, x1.to(device), t1.to(device), order=1).to(device) * test1_dx.to(device) *  dx * dt \
          + dfdt(pinn, x1, t1).to(device) * test1.to(device) * dx * dt \
          + epsilon * u0_dx1_t1.to(device) * test1_dx.to(device)  * dx * dt \
          + epsilon * u0_x1_dt1.to(device) * test1_dt.to(device)  * dx * dt \
          + u0_x1_dt1.to(device) * test1.to(device) * dx * dt
        loss_tmp_weak2 = \
          + epsilon * dfdt(pinn, x2.to(device), t2.to(device), order=1).to(device) * test2_dt.to(device) * dx * dt \
          + epsilon * dfdx(pinn, x2.to(device), t2.to(device), order=1).to(device) * test2_dx.to(device) *  dx * dt \
          + dfdt(pinn, x2, t2).to(device) * test2.to(device) * dx * dt \
          + epsilon * u0_dx2_t2.to(device) * test2_dx.to(device)  * dx * dt \
          + epsilon * u0_x2_dt2.to(device) * test2_dt.to(device)  * dx * dt \
          + u0_x2_dt2.to(device) * test2.to(device) * dx * dt
        loss_tmp_weak3 = \
          + epsilon * dfdt(pinn, x3.to(device), t3.to(device), order=1).to(device) * test3_dt.to(device) * dx * dt \
          + epsilon * dfdx(pinn, x3.to(device), t3.to(device), order=1).to(device) * test3_dx.to(device) *  dx * dt \
          + dfdt(pinn, x3, t3).to(device) * test3.to(device) * dx * dt \
          + epsilon * u0_dx3_t3.to(device) * test3_dx.to(device)  * dx * dt \
          + epsilon * u0_x3_dt3.to(device) * test3_dt.to(device)  * dx * dt \
          + u0_x3_dt3.to(device) * test3.to(device) * dx * dt
        loss_tmp_weak4 = \
          + epsilon * dfdt(pinn, x4.to(device), t4.to(device), order=1).to(device) * test4_dt.to(device) * dx * dt \
          + epsilon * dfdx(pinn, x4.to(device), t4.to(device), order=1).to(device) * test4_dx.to(device) *  dx * dt \
          + dfdt(pinn, x4, t4).to(device) * test4.to(device) * dx * dt \
          + epsilon * u0_dx4_t4.to(device) * test4_dx.to(device)  * dx * dt \
          + epsilon * u0_x4_dt4.to(device) * test4_dt.to(device)  * dx * dt \
          + u0_x4_dt4.to(device) * test4.to(device) * dx * dt
        final_loss+=  \
          (loss_tmp_weak1+loss_tmp_weak2+loss_tmp_weak3+loss_tmp_weak4).sum().pow(2) * constant

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

    final_loss = \
        weight_f * interior_loss(pinn, x, t, epsilon)

#PRINTING
    number = 40
    number2 = 1600

    x_domain_loc = [0.0, length]
    t_domain_loc = [0.0, total_time]
    h = 1/40
    device = x.device
    #midpoints
    x_raw_loc = torch.linspace(x_domain_loc[0]-0.5*h, x_domain_loc[1]+0.5*h, steps=40, requires_grad=True)
    t_raw_loc = torch.linspace(t_domain_loc[0]-0.5*h, t_domain_loc[1]+0.5*h, steps=40, requires_grad=True)
    grids_loc = torch.meshgrid(x_raw_loc, t_raw_loc, indexing="ij")
    # https://www.minhnguyencae.info/2d-gaussian-quadrature/
    point = (math.sqrt(3)/3.0)*0.5 # (3^(1/3)/3)*0.5*h (we map [-1,1] into [0,h]), weights=1
    x_loc1 = grids_loc[0].flatten().reshape(-1, 1).to(device)-point*h
    t_loc1 = grids_loc[1].flatten().reshape(-1, 1).to(device)-point*h
    x_loc2 = grids_loc[0].flatten().reshape(-1, 1).to(device)+point*h
    t_loc2 = grids_loc[1].flatten().reshape(-1, 1).to(device)-point*h
    x_loc3 = grids_loc[0].flatten().reshape(-1, 1).to(device)-point*h
    t_loc3 = grids_loc[1].flatten().reshape(-1, 1).to(device)+point*h
    x_loc4 = grids_loc[0].flatten().reshape(-1, 1).to(device)+point*h
    t_loc4 = grids_loc[1].flatten().reshape(-1, 1).to(device)+point*h
    z_loc1 = f(pinn.to(device), x_loc1, t_loc1)+shift(x_loc1,t_loc1)
    z_loc2 = f(pinn.to(device), x_loc2, t_loc2)+shift(x_loc2,t_loc2)
    z_loc3 = f(pinn.to(device), x_loc3, t_loc3)+shift(x_loc2,t_loc3)
    z_loc4 = f(pinn.to(device), x_loc4, t_loc4)+shift(x_loc2,t_loc4)

    #dzdx
    dz_dx1 = dfdx(pinn.to(device), x_loc1, t_loc1, order=1)
    dz_dx2 = dfdx(pinn.to(device), x_loc2, t_loc2, order=1)
    dz_dx3 = dfdx(pinn.to(device), x_loc3, t_loc3, order=1)
    dz_dx4 = dfdx(pinn.to(device), x_loc4, t_loc4, order=1)
    dz_dx = (dz_dx1+dz_dx2+dz_dx3+dz_dx4)
#    #dfdx jest duze w pierwszym i ostatnim punkcie number2-1
#    for i in range(0,number2):
#      if dz_dx[i]>1:
#        print(dz_dx[i])
#        print(i)
#      if dz_dx[i]<-1:
#        print(dz_dx[i])
#        print(i)
#    dz_dx0 =dz_dx[0]
#    dz_dx9999 =dz_dx[number2-1]
#    dz_dx_norm = ((dz_dx.pow(2).sum()-dz_dx0*dz_dx0-dz_dx9999*dz_dx9999)/number2)
    dz_dx_norm = dz_dx.pow(2).sum()/number2
#    print(f'dz_dx_norm:{dz_dx_norm.item():.5f}')

    #dzdt
    dz_dt1 = dfdt(pinn.to(device), x_loc1, t_loc1, order=1)
    dz_dt2 = dfdt(pinn.to(device), x_loc2, t_loc2, order=1)
    dz_dt3 = dfdt(pinn.to(device), x_loc3, t_loc3, order=1)
    dz_dt4 = dfdt(pinn.to(device), x_loc4, t_loc4, order=1)
    dz_dt = (dz_dt1+dz_dt2+dz_dt3+dz_dt4)
#    #dfdt jest duze
#    for i in range(0,number2):
#      if dz_dt[i]>1:
#        print(dz_dt[i])
#        print(i)
#      if dz_dt[i]<-1:
#        print(dz_dt[i])
#        print(i)
#    dz_dt0 =dz_dt[0]
#    dz_dt9999 =dz_dt[number2-1]
#    dz_dt_norm = ((dz_dt.pow(2).sum()-dz_dt9999*dz_dt9999-dz_dt0*dz_dt0)/number2)
    dz_dt_norm = dz_dt.pow(2).sum()/number2
#    print(f'dz_dt_norm:{dz_dt_norm.item():.5f}')

    #exact_dx
    exact_dx1 = exact_solution_dx(x_loc1,t_loc1, epsilon)
    exact_dx2 = exact_solution_dx(x_loc2,t_loc2, epsilon)
    exact_dx3 = exact_solution_dx(x_loc3,t_loc3, epsilon)
    exact_dx4 = exact_solution_dx(x_loc4,t_loc4, epsilon)
    exact_dx = (exact_dx1+exact_dx2+exact_dx3+exact_dx4)
    exact_dx_norm = exact_dx.pow(2).sum()/number2
#    print(f'exact_dx_norm:{exact_dx_norm:.5f}')

    #exact_dt
    exact_dt1 = exact_solution_dt(x_loc1,t_loc1, epsilon)
    exact_dt2 = exact_solution_dt(x_loc2,t_loc2, epsilon)
    exact_dt3 = exact_solution_dt(x_loc3,t_loc3, epsilon)
    exact_dt4 = exact_solution_dt(x_loc4,t_loc4, epsilon)
    exact_dt = (exact_dt1+exact_dt2+exact_dt3+exact_dt4)
    exact_dt_norm = exact_dt.pow(2).sum()/number2
#    print(f'exact_dt_norm:{exact_dt_norm:.5f}')

    #shift_dx
    shift_dx_all1 = shift_dx(x_loc1,t_loc1)
    shift_dx_all2 = shift_dx(x_loc2,t_loc2)
    shift_dx_all3 = shift_dx(x_loc3,t_loc3)
    shift_dx_all4 = shift_dx(x_loc4,t_loc4)
    shift_dx_all = (shift_dx_all1+shift_dx_all2+shift_dx_all3+shift_dx_all4)
    shift_dx_norm = (shift_dx_all.pow(2).sum()/number2)
#    print(f'shift_dx_norm:{shift_dx_norm:.5f}')

    #shift_dt
    shift_dt_all1 = shift_dt(x_loc1,t_loc1)
    shift_dt_all2 = shift_dt(x_loc2,t_loc2)
    shift_dt_all3 = shift_dt(x_loc3,t_loc3)
    shift_dt_all4 = shift_dt(x_loc4,t_loc4)
    shift_dt_all = (shift_dt_all1+shift_dt_all2+shift_dt_all3+shift_dt_all4)
    shift_dt_norm = (shift_dt_all.pow(2).sum()/number2)
#    print(f'shift_dt_norm:{shift_dt_norm:.5f}')

    VM_z_norm = epsilon*math.sqrt((dz_dx_norm+dz_dt_norm-exact_dx_norm-exact_dt_norm+shift_dx_norm+shift_dt_norm)**2)
    VM_exact_norm = epsilon*math.sqrt((exact_dx_norm+exact_dt_norm)**2)

#    print(f'VM_z_norm:{VM_z_norm:.5f}')
#    print(f'VM_exact_norm:{VM_exact_norm:.5f}')
    VM_norm = VM_z_norm / VM_exact_norm
#    print(f'VM_norm:{VM_norm:.5f}')

    z1 = f(pinn.to(device), x_loc1, t_loc1)+shift(x_loc1,t_loc1)-exact_solution(x_loc1,t_loc1, epsilon)
    z2 = f(pinn.to(device), x_loc2, t_loc2)+shift(x_loc2,t_loc2)-exact_solution(x_loc2,t_loc2, epsilon)
    z3 = f(pinn.to(device), x_loc3, t_loc3)+shift(x_loc3,t_loc3)-exact_solution(x_loc3,t_loc3, epsilon)
    z4 = f(pinn.to(device), x_loc4, t_loc4)+shift(x_loc4,t_loc4)-exact_solution(x_loc4,t_loc4, epsilon)
    z = (z1+z2+z3+z4)
    exact1 = exact_solution(x_loc1,t_loc1, epsilon)
    exact2 = exact_solution(x_loc2,t_loc2, epsilon)
    exact3 = exact_solution(x_loc3,t_loc3, epsilon)
    exact4 = exact_solution(x_loc4,t_loc4, epsilon)
    exact = (exact1+exact2+exact3+exact4)
    L2_exact_norm = epsilon*math.sqrt(exact.pow(2).sum())/number2 #the values are (0,1) we average them
    L2_z_norm = epsilon*math.sqrt(z.pow(2).sum())/number2
    L2_norm = L2_z_norm/L2_exact_norm
#    print(f'L2_z_norm:{L2_z_norm:.5f}')
#    print(f'L2_exact_norm:{L2_exact_norm:.5f}')
#    print(f'L2_norm:{L2_norm:.5f}')

    #loss
    interior_loss_val = interior_loss(pinn, x, t, epsilon)

    with open('OutputPlots.txt', 'a') as file_f:
      print(f"||u_exact||VM: {VM_exact_norm:.7f} ||u_NN-u_exact||VM: {VM_z_norm:.7f} ||u_NN-u_exact||VM/||u_exact||VM: {VM_norm:.7f} ||u_exact||L2: {L2_exact_norm:.7f} ||u_NN-u_exact||L2: {L2_z_norm:.7f} ||u_NN-u_exact||L2/||u_exact||L2: {L2_norm:.7f} sqrt(InteriorLoss)/||u_exact||VM: {float(math.sqrt(interior_loss_val)/VM_exact_norm):.7f} sqrt(InteriorLoss)/||u_exact||L2: {float(math.sqrt(interior_loss_val)/L2_exact_norm):.7f} sqrt(InteriorLoss): {float(math.sqrt(interior_loss_val)):.7f} InteriorLoss: {float(interior_loss_val):.7f} ", file=file_f)


    if not verbose:
        return final_loss
    else:
        return final_loss, interior_loss(pinn, x, t), initial_loss(pinn, x_init, t), finito_loss(pinn, x_init, t), bottom_loss(pinn, x_init, t), top_loss(pinn, x_init, t)