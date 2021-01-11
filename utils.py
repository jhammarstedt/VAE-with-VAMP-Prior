import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import time

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


def val_loop(val_loader, model, writer, epoch, plot=False, directory=''):
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
                if not os.path.exists(directory + 'reconstruction/'):
                    os.makedirs(directory + 'reconstruction/')
                plot_images(x.cpu().detach().numpy()[:9], directory + 'reconstruction/', 'real')
            reconstruction = model.reconstruct(x)
            plot_images(reconstruction.cpu().detach().numpy()[:9], directory + 'reconstruction/', str(epoch))
    writer.flush()

    # calculate final loss
    val_loss /= len(val_loader)  # loss function already averages over batch size
    val_re /= len(val_loader)  # re already averages over batch size
    val_kl /= len(val_loader)  # kl already averages over batch size
    return val_loss, val_re, val_kl


def test_loop(test_loader, model, directory=''):
    test_data = []
    for data, _ in test_loader:
        test_data.append(data)

    test_data = torch.cat(test_data, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = test_data.to(device)

    print('Plot real images, reconstructions and sampled images')
    reconstruction = model.reconstruct(test_data[:25])
    plot_images(test_data[:25].cpu().detach().numpy(), directory + 'test_img/', 'real', size_x=5, size_y=5)
    plot_images(reconstruction.cpu().detach().numpy(), directory + 'test_img/', 'reconstructions', size_x=5, size_y=5)
    sampled = model.sample(25)
    plot_images(sampled.cpu().detach().numpy(), directory + 'test_img/', 'sampled', size_x=5, size_y=5)
    if args['prior'] == 'vamp':
        pseudo_inputs = model.pseudo_mapper(model.pseudo_input)
        plot_images(pseudo_inputs[:25].cpu().detach().numpy(), directory + 'test_img/', 'pseudoinputs', size_x=5, size_y=5)


    ll_test_start = time.time()
    ll_test = model.compute_LL(test_data, 1000, 100)
    ll_test_end = time.time()
    print('Log-likelihood on test data {} in {}min'.format(ll_test, (ll_test_end - ll_test_start) / 60))
    return ll_test


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



def plot_images(x_sample, directory, name, size_x=3, size_y=3):

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

    plt.savefig(directory + name + '.png', bbox_inches='tight')
    plt.close(fig)
