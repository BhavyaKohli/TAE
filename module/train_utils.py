from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

device = 'cpu' if torch.cuda.is_available() else 'cpu'

class ProgressBar():
    def __init__(self, epochs, ncols=100, verbose=1):
        self.bar = tqdm(range(1,epochs+1), ncols=ncols, disable=not verbose, 
                        bar_format="Epoch: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}{postfix}]")
        self.bar.set_postfix_str(f"loss: -, es: -")

    def update(self, loss, es=None):
        post_str = f"loss: {loss:.4f}"
        post_str = post_str + f", es: {es}" if es is not None else post_str
        self.bar.set_postfix_str(post_str)

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

class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.targets = Y

    def get_dataloader(self, batch_size, shuffle):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def __len__(self):
        return len(self.data)

class Autoencoder(nn.Module):
    def __init__(self, in_features, embed):
        super().__init__()
        self.enc = nn.Linear(in_features, embed, bias=False)
        self.dec = nn.Linear(embed, in_features, bias=False)
    
    def forward(self, x, return_embed=False):
        embed = self.enc(x)
        recon = self.dec(embed)
        return (embed, recon) if return_embed else recon 

def train_ae(autoencoder, dataloader, optimizer, epochs, verbose=0, pbar_ncols=75, early_stopping=10):
    criterion = nn.MSELoss()
    if early_stopping is False: early_stopping = np.inf

    pbar = ProgressBar(epochs, ncols=pbar_ncols, verbose=verbose)
    es = EarlyStopping()

    losses = []
    for epoch in pbar.bar:
        if es.es >= early_stopping: break

        batch_losses = []
        for x, y in dataloader:
            x, y = x.to(device).float(), y.to(device).float()

            recon = autoencoder(x)
            loss = criterion(recon, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        
        losses.append(np.mean(batch_losses))
        es.update(losses[-1])
        if verbose: pbar.update(losses[-1], es.es)
    
    return autoencoder, losses
