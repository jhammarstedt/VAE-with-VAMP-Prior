from torch.autograd import Variable

def train_vae(train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        x = data

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        loss, RE, KL = model.get_loss(x)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.item()
        train_re += -RE.item()
        train_kl += KL.item()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl

