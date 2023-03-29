import torch
def data_loss_factory(
        loss_fn, 
        use_var_list = 'default', 
        weight = 1.0,
        device = 'default'
    ):
    if device == 'default':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if use_var_list == 'default':
        def data_loss(model, data):
            x_real,y_real = data
            y_pred = model(x_real)
            loss = loss_fn(y_pred, y_real)
            return weight * loss
        return data_loss
    else:
        use_var_list = torch.tensor(use_var_list,device=device)
        def data_loss(model, data):
            x_real,y_real = data
            y_pred = model(x_real)
            loss = loss_fn(y_pred[:,use_var_list], y_real)
            return weight * loss
        return data_loss