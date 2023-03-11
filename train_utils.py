from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

class ProgressBar():
    def __init__(self, epochs, ncols=100):
        self.bar = tqdm(range(1,epochs+1), ncols=ncols)
        self.bar.set_description_str(f"Epoch {0}")
        self.bar.set_postfix_str(f"loss: -, es: -")

    def update(self, epoch, loss, es=None):
        post_str = f"loss: {loss:.4f}"
        post_str = post_str + f", es: {es}" if es is not None else post_str
        self.bar.set_postfix_str(post_str)
        self.bar.set_description_str(f"Epoch {epoch}")

def train_step(model, criterion, optimizer, x, y, centers=None):
    x, y = x.cuda().float(), y.cuda().float()

    embed, recon = model(x) if centers is None else model(x, centers)
    loss = criterion(recon, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()    

class EarlyStopping():
    def __init__(self):
        self.es = 0
        self.loss = 0
    
    def update(self, latest_loss):
        if latest_loss >= self.loss:
            self.es += 1
            self.loss = latest_loss
        else:
            self.es = 0
            self.loss = latest_loss


def train_ae(autoencoder, dataloader, optimizer, epochs, lr=0.01, verbose=0):
    criterion = nn.MSELoss()

    pbar = ProgressBar(epochs, ncols=100) if verbose else range(epochs)
    pbar_ = pbar.bar if verbose else pbar
    es = EarlyStopping()

    losses = []
    for i in pbar_:
        if es.es >= 10: break

        batch_losses = []
        for x, y in dataloader:
            batch_losses.append(
                train_step(
                    autoencoder, criterion, optimizer,
                    x, y
                )
            )
        losses.append(np.mean(batch_losses))
        es.update(losses[-1])
        if verbose: pbar.update(i, losses[-1])
    
    return autoencoder, losses
