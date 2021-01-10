from model import VAE_model
from load_data import load_dataset
from torch.optim import Adam
from utils import train_loop, val_loop
from torch.utils.tensorboard import SummaryWriter

args = {
    'batch_size': 100,
    'input_size': [1, 28, 28],
    'epochs': 20,
    'lr': 0.0001,
    'hidden_dims': 100,
    'latent_dims': 10,
    'input_type': 'continuous',  # ['binary','continuous']
    'prior': 'vamp',  # ['vamp','standard']
    'psudo_inp': 200,  # ignore if standard
    'dataset': 'dynamicMnist',  # ['dynamicMnist', 'fashionMnist', 'omniglot']
}

train_loader, val_loader, test_loader, input_size = load_dataset(args)

writer = SummaryWriter()

train_loss_history = []
train_re_history = []
train_kl_history = []

model = VAE_model(input_size=input_size[1], args=args)
optimizer = Adam(model.parameters(), lr=args['lr'])

for epoch in range(1, args['epochs'] + 1):
    model, tr_loss_e, tr_re_e, tr_kl_e = train_loop(train_loader=train_loader, model=model, optimizer=optimizer, writer=writer, epoch=epoch)
    val_loss, val_re, val_kl = val_loop(val_loader, model, writer, epoch, plot=True)
    train_loss_history.append(tr_loss_e)
    train_re_history.append(tr_re_e)
    train_kl_history.append(tr_kl_e)
    print("Epoch {},\t train_loss: {:.2f}\t(RL = {:.2f},\tKL: {:.2f})\t val_loss {:.2f}\t(RL = {:.2f},\tKL: {:.2f})".format(epoch, tr_loss_e, tr_re_e, tr_kl_e, val_loss, val_re, val_kl))

writer.close()