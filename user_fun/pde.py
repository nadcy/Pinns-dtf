import torch

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

def diff(y, x, degree=1):
    if degree == 1:
        return torch.autograd.grad(y, x,
                               grad_outputs=torch.ones_like(y),
                               create_graph=True)[0]
    else:
        dydx = diff(y,x,degree-1)
        return torch.autograd.grad(dydx, x,
                               grad_outputs=torch.ones_like(dydx),
                               create_graph=True)[0]

def ico_2D_static_factory(loss_fn,MU):
    def pde_loss(model, data):
        x_in,y_real = data
        x_in.requires_grad=True

        x = x_in[:,[0]]
        y = x_in[:,[1]]
        x_use = torch.cat((x,y),dim = 1)
        U = model(x_use)
        p = U[:,[0]]
        u = U[:,[1]]
        v = U[:,[2]]

        dudx = diff(u,x)
        dudy = diff(u,y)
        dvdx = diff(v,x)
        dvdy = diff(v,y)
        dpdx = diff(p,x)
        dpdy = diff(p,y)

        du2dx2 = diff(dudx,x)
        du2dy2 = diff(dudy,y)
        dv2dx2 = diff(dvdx,x)
        dv2dy2 = diff(dvdy,y)

        eq1 = u * dudx + v * dudy + dpdx - MU * (du2dx2 + du2dy2)
        eq2 = u * dvdx + v * dvdy + dpdy - MU * (dv2dx2 + dv2dy2)
        eq3 = dudx + dvdy
        loss_val = loss_fn(eq1, y_real[:,[0]]) + loss_fn(eq2, y_real[:,[1]]) + \
            3*loss_fn(eq3, y_real[:,[2]])
        return loss_val
    return pde_loss

def ico_time_pde_loss_factory(loss_fn,MU):

    def ico_time_pde_loss(model,data):
        x_in,y_real = data
        x_in.requires_grad=True

        x = x_in[:,[0]]
        y = x_in[:,[1]]
        t = x_in[:,[2]]
        x_use = torch.cat((x,y,t),dim = 1)

        U = model(x_use)
        p = U[:,[0]]
        u = U[:,[1]]
        v = U[:,[2]]

        # dudt = diff(u,t)
        # dvdt = diff(v,t)
        dudx = diff(u,x)
        # dudy = diff(u,y)
        # dvdx = diff(v,x)
        dvdy = diff(v,y)
        # dpdx = diff(p,x)
        # dpdy = diff(p,y)

        # du2dx2 = diff(dudx,x)
        # du2dy2 = diff(dudy,y)
        # dv2dx2 = diff(dvdx,x)
        # dv2dy2 = diff(dvdy,y)

        # eq1 = u * dudx + v * dudy + dpdx - MU * (du2dx2 + du2dy2) 
        # eq2 = u * dvdx + v * dvdy + dpdy - MU * (dv2dx2 + dv2dy2)
        # eq3 = dudx + dvdy

        # def nor(a,b):
        #     return (a-a.mean())/a.std(),(b-b.mean())/b.std()
        
        # nor_eq1,nor_dudt = nor(eq1,dudt)
        # nor_eq2,nor_dvdt = nor(eq2,dvdt)
        # nor_dudx,nor_dvdy = nor(dudx,dvdy)

        # loss_val = loss_fn(nor_eq1,nor_dudt) + \
        #     + loss_fn(nor_eq2,nor_dvdt) +\

        
        loss_val = loss_fn(dudx,-dvdy)
        return loss_val
    
    return ico_time_pde_loss

