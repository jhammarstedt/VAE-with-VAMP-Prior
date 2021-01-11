import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def train_loop(train_loader, model, optimizer, writer, epoch):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for x, y in train_loader:
        x = x.to(device)

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        if epoch > 100: #warmup period
            print('Ending Warmup')
            loss, RE, KL = model.get_loss(x,warmup=False)
        else:
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


def val_loop(val_loader, model, writer, epoch, plot=False, dir=''):
    # set loss to 0
    val_loss = 0
    val_re = 0
    val_kl = 0
    # set model in training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for id, (x, _) in enumerate(val_loader):
        x = x.to(device)
        # loss evaluation (forward pass)
        loss, RE, KL = model.get_loss(x)

        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("RE", RE, epoch)
        writer.add_scalar("KL", KL, epoch)

        val_loss += loss.item()
        val_re += RE.item()
        val_kl += KL.item()
        if plot and id == 1:
            if epoch == 1:
                if not os.path.exists(dir + 'reconstruction/'):
                    os.makedirs(dir + 'reconstruction/')
                plot_images(x.cpu().detach().numpy()[:9], dir + 'reconstruction/', 'real')
            _, _, reconstruction, _, _, _, _ = model.forward(x)
            plot_images(reconstruction.cpu().detach().numpy()[:9], dir + 'reconstruction/', str(epoch))
    writer.flush()

    # calculate final loss
    val_loss /= len(val_loader)  # loss function already averages over batch size
    val_re /= len(val_loader)  # re already averages over batch size
    val_kl /= len(val_loader)  # kl already averages over batch size
    return val_loss, val_re, val_kl

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

def plot_images(x_sample, dir, name, size_x=3, size_y=3):

    fig = plt.figure(figsize=(size_x, size_y))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape((28, 28))
        plt.imshow(sample)

    plt.savefig(dir + name + '.png', bbox_inches='tight')
    plt.close(fig)
