from model import VAE_model
from load_data import load_dataset
import torch
from torch.optim import Adam
from utils import train_loop, val_loop, test_loop
from torch.utils.tensorboard import SummaryWriter

import time
import os
import shutil

args = {
    'batch_size': 100,
    'input_size': [1, 28, 28],
    'epochs': 10,
    'lr': 0.0001,
    'hidden_dims': 100,
    'latent_dims': 10,
    'input_type': 'continuous',  # ['binary','continuous']
    'prior': 'vamp',  # ['vamp','standard']
    'psudo_inp': 200,  # ignore if standard
    'dataset': 'dynamicMnist',  # ['dynamicMnist', 'fashionMnist', 'omniglot']
    'train': True,
    'test': True,
}


def run_experiment(args):
    # load data
    train_loader, val_loader, test_loader, input_size = load_dataset(args)

    run_name = '{}_{}_{}_{}'.format(args['dataset'], args['prior'], args['psudo_inp'], args['latent_dims'])
    model_name = run_name + '.model'
    model = VAE_model(input_size=input_size[1], args=args)

    # create log directory
    if os.path.exists(run_name):
        shutil.rmtree(run_name)
    os.mkdir(run_name)

    log_file = run_name + '/results.txt'
    with open(log_file, 'a') as f:
        print(args, file=f)
    print(args)
    if args['train']:
        optimizer = Adam(model.parameters(), lr=args['lr'])
        writer = SummaryWriter()

        train_loss_history = []
        train_re_history = []
        train_kl_history = []

        val_loss_history = []
        val_re_history = []
        val_kl_history = []

        best_val_loss = 1e10
        for epoch in range(1, args['epochs'] + 1):
            start = time.time()
            model, tr_loss_e, tr_re_e, tr_kl_e = train_loop(train_loader=train_loader, model=model, optimizer=optimizer, writer=writer, epoch=epoch)
            val_loss_e, val_re_e, val_kl_e = val_loop(val_loader, model, writer, epoch, plot=True, directory=run_name + '/')

            train_loss_history.append(tr_loss_e)
            train_re_history.append(tr_re_e)
            train_kl_history.append(tr_kl_e)

            val_loss_history.append(val_loss_e)
            val_re_history.append(val_re_e)
            val_kl_history.append(val_kl_e)
            end = time.time()
            print(
                "Epoch {}\ttime {:.2f}s,\t train_loss: {:.2f}\t(RL = {:.2f},\tKL: {:.2f})\t val_loss {:.2f}\t(RL = {:.2f},\tKL: {:.2f})".format(
                    epoch, end - start, tr_loss_e, tr_re_e, tr_kl_e, val_loss_e, val_re_e, val_kl_e))

            with open(log_file, 'a') as f:
                print("Epoch {}\ttime {:.2f}s,\t train_loss: {:.2f}\t(RL = {:.2f},\tKL: {:.2f})\t val_loss {:.2f}\t(RL = {:.2f},\tKL: {:.2f})".format(epoch, end - start, tr_loss_e, tr_re_e, tr_kl_e, val_loss_e, val_re_e, val_kl_e), file=f)

            if val_loss_e < best_val_loss:
                best_val_loss = val_loss_e
                print('Saving model with {} validation loss'.format(best_val_loss))
                torch.save(model.state_dict(), run_name + '/' + model_name)

        writer.close()

    if args['test']:
        os.mkdir(run_name + '/test_img/')
        model.load_state_dict(torch.load(run_name + '/' + model_name))
        ll_test = test_loop(test_loader, model, args, directory=run_name + '/')
        with open(log_file, 'a') as f:
            print('Model Log-likelihood on test set is {}'.format(ll_test), file=f)

    shutil.make_archive(run_name, 'tar', run_name)
    return model


if __name__ == '__main__':
    print(args)
    run_experiment(args)
