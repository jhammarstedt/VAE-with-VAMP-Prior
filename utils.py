from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt


def train_vae(train_loader, model, optimizer, writer, epoch):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    for data, target in train_loader:
        data, target = data, target
        x = data

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        loss, RE, KL = model.get_loss(x)

        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("RE", RE, epoch)
        writer.add_scalar("KL", KL, epoch)

        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.item()
        train_re += RE.item()
        train_kl += KL.item()
    writer.flush()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


# def evaluation():
#
#
# def generate_plot(x_label, ):
#     fig = plt.figure()
#     plt.xlabel('Log-likelihood value')
#     plt.ylabel('Probability')
#
#
#     plt.grid(True)
#     # plt.savefig(f"{}")
#     plt.show()