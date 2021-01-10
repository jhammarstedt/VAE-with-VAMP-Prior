from model import VanillaVAE
from load_data import load_dynamic_mnist
from torch.optim import Adam
from utils import train_vae
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

args = {
    'batch_size': 100,
    'input_size': [1, 28, 28],
    'epochs': 1,
    'lr': 0.0001,
    'input_type': 'continuous',
}

train_loader, val_loader, test_loader, input_size = load_dynamic_mnist(args['batch_size'])

# data, target = next(iter(train_loader))
#
# plt.imshow(data[0].reshape(28, 28))
# plt.show()

writer = SummaryWriter()

train_loss_history = []
train_re_history = []
train_kl_history = []

model = VanillaVAE(input_size=input_size[1], hidden_dims=100, latent_dims=10, args=args)
optimizer = Adam(model.parameters(), lr=args['lr'])

for epoch in range(1, args['epochs'] + 1):
    model, tr_loss_e, tr_re_e, tr_kl_e = train_vae(train_loader=train_loader, model=model, optimizer=optimizer, writer=writer, epoch=epoch)
    train_loss_history.append(tr_loss_e)
    train_re_history.append(tr_re_e)
    train_kl_history.append(tr_kl_e)
    print("Epoch {},\t loss: {},\t reconstruction loss: {:.3f},\t KL: {:.3f}".format(epoch, tr_loss_e, tr_re_e, tr_kl_e))

writer.close()