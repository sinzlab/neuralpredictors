import torch


def train_step(model, loss_fn, optim, batch, device):
    # put model in train mode
    model.train()
    model.to(device)

    #  run one forward + backward step
    # clear gradient
    optim.zero_grad()
    # inputs and targets
    inputs, targets = batch[0], batch[1]
    # move data to DEVICE
    inputs = inputs.to(device)
    targets = targets.to(device)
    # forward
    likelihood = model(inputs)
    # calc loss
    loss = loss_fn(likelihood, targets)
    # backward
    loss.backward()
    # optimization step
    optim.step()


def compare_vars(vars_change, model, loss_fn, optim, batch, device, params=None):
    if params is None:
        # get a list of params
        params = [np for np in model.named_parameters()]

    # take a copy
    initial_params = [(name, p.clone()) for (name, p) in params]

    # run a training step
    train_step(model, loss_fn, optim, batch, device)

    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            if vars_change:
                assert not torch.equal(p0.to(device), p1.to(device))
            else:
                assert torch.equal(p0.to(device), p1.to(device))
        except AssertionError:
            raise ValueError(
                "{var_name} {msg}".format(var_name=name, msg="did not change!" if vars_change else "changed!")
            )
