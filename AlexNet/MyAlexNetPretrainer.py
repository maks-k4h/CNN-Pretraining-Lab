import torch
from torch import nn
from AlexNet.MyAlexNet import AlexNetMini
from itertools import chain
import statistics


class AlexNetPretrainer(nn.Module):
    """Pretrainer for AlexNet Mini."""

    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.Conv1Transpose = nn.ConvTranspose2d(48, 3, kernel_size=11, stride=4, padding=2)

        self.Conv2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.Conv2Transpose = nn.ConvTranspose2d(128, 48, kernel_size=5, padding=2)

        self.Conv3 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.Conv3Transpose = nn.ConvTranspose2d(192, 128, kernel_size=3, padding=1)

        self.Conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.Conv4Transpose = nn.ConvTranspose2d(192, 192, kernel_size=3, padding=1)

        self.Conv5 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.Conv5Transpose = nn.ConvTranspose2d(128, 192, kernel_size=3, padding=1)

    def encode(self, x, stage: int, return_indices=False):
        MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=return_indices)
        if stage == 1:
            x = self.Conv1(x)
            x = nn.ReLU()(x)
            res = MaxPool(x)
        elif stage == 2:
            x = self.Conv2(x)
            x = nn.ReLU()(x)
            res = MaxPool(x)
        elif stage == 3:
            x = self.Conv3(x)
            res = nn.ReLU()(x)
        elif stage == 4:
            x = self.Conv4(x)
            res = nn.ReLU()(x)
        elif stage == 5:
            x = self.Conv5(x)
            x = nn.ReLU()(x)
            res = MaxPool(x)
        else:
            raise "No such stage."

        return res

    def decode(self, x, stage: int, indices=None):
        MaxUnpool = nn.MaxUnpool2d(kernel_size=3, stride=2,)
        if stage == 1:
            x = MaxUnpool(x, indices=indices)
            x = self.Conv1Transpose(x)
        elif stage == 2:
            x = MaxUnpool(x, indices=indices)
            x = self.Conv2Transpose(x)
        elif stage == 3:
            x = self.Conv3Transpose(x)
        elif stage == 4:
            x = self.Conv4Transpose(x)
        elif stage == 5:
            x = MaxUnpool(x, indices=indices)
            x = self.Conv5Transpose(x)
        else:
            raise "No such stage."

        return x

    def forward(self, x):
        x, i1 = self.encode(x, 1, return_indices=True)
        x, i2 = self.encode(x, 2, return_indices=True)
        x = self.encode(x, 3)
        x = self.encode(x, 4)
        x, i5 = self.encode(x, 5, return_indices=True)

        print('Code size:', x.shape)

        x = self.decode(x, 5, indices=i5)
        x = self.decode(x, 4)
        x = self.decode(x, 3)
        x = self.decode(x, 2, indices=i2)
        x = self.decode(x, 1, indices=i1)
        return x

    def pretrain(self, dataloader, stage, epochs, lr, momentum, wd, unsupervised_loss=nn.L1Loss(reduction='mean'), unpack=False):

        print('Pretraining the', stage, 'th layer of AlexNet Mini.')
        print('Epochs:', epochs)
        print('Learning rate:', lr)
        print('Momentum:', momentum)
        print('Weight decay:', wd)

        # setting up the optimizer
        parameters = None
        if stage == 1:
            parameters = chain(self.Conv1.parameters(), self.Conv1Transpose.parameters())
        elif stage == 2:
            parameters = chain(self.Conv2.parameters(), self.Conv2Transpose.parameters())
        elif stage == 3:
            parameters = chain(self.Conv3.parameters(), self.Conv3Transpose.parameters())
        elif stage == 4:
            parameters = chain(self.Conv4.parameters(), self.Conv4Transpose.parameters())
        elif stage == 5:
            parameters = chain(self.Conv5.parameters(), self.Conv5Transpose.parameters())

        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)

        N = len(dataloader)
        Nb = max(1, N // 16)

        for epoch in range(epochs):
            print('Epoch', epoch + 1)
            epoch_losses = []
            batches_losses = []

            for bn, batch in enumerate(dataloader):
                if unpack:
                    batch = batch[0]

                # reporting the number of batches done
                if (bn + 1) % Nb == 0:
                    print('[{:6} | {:6}] loss: {}'.format(bn + 1, N, statistics.mean(batches_losses)))
                    batches_losses = []

                # getting the output from prior layers
                for i in range(1, stage):
                    batch = self.encode(batch, i)

                # generating the code and the reconstruction and estimating the loss
                res = self.encode(batch, stage, return_indices=True)
                if stage in {1, 2, 5}:
                    recon = self.decode(res[0], stage, indices=res[1])
                else:
                    recon = self.decode(res, stage)
                loss = unsupervised_loss(batch, recon)

                # tracking the loss
                epoch_losses.append(float(loss))
                batches_losses.append(float(loss))

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch loss:', statistics.mean(epoch_losses))

    def appy_weights(self, model: AlexNetMini):
        model.features.load_state_dict({
            '0.weight': self.Conv1.weight,
            '0.bias': self.Conv1.bias,
            '3.weight': self.Conv2.weight,
            '3.bias': self.Conv2.bias,
            '6.weight': self.Conv3.weight,
            '6.bias': self.Conv3.bias,
            '8.weight': self.Conv4.weight,
            '8.bias': self.Conv4.bias,
            '10.weight': self.Conv5.weight,
            '10.bias': self.Conv5.bias,
        })
