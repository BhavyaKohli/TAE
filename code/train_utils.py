from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

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
            x, y = x.to(autoencoder.device).float(), y.to(autoencoder.device).float()

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

def train_tae(tae, X, Y, epochs, lr, batch_size, warmup=0.3, warmup_optim=torch.optim.Adagrad, warmup_lr=0.1, verbose=True, grad_clip=5):
    clust_assign = tae.assign_centers_to_data(Y, one_hot=True)
    tae.centers = tae.update_centers(Y, clust_assign)
    
    optimizer = warmup_optim(tae.parameters(), warmup_lr)
    warmup_dataloader = GenericDataset(X, Y).get_dataloader(batch_size=1, shuffle=False)
    
    warmup_epochs = int(warmup*epochs) if warmup <= 1 else warmup

    if verbose: print(f"PHASE 1: Warmup — {warmup_epochs}/{epochs}")
    pbar_warmup = ProgressBar(warmup_epochs, 75, verbose)
    warmup_losses = []
    for epoch in pbar_warmup.bar:
        batch_losses = []
        new_clust_assign = torch.tensor([])
        for n, (x,y) in enumerate(warmup_dataloader):
            x, y = x.to(tae.device), y.to(tae.device)

            loss, new_assignment = tae.compute_loss_warmup(x, y, clust_assign[:,n:n+1])
            new_clust_assign = torch.cat((new_clust_assign, new_assignment), dim=1)

            # if new_assignment.argmax() != clust_assign[:, sl_clust].argmax(): print(f"updated {samp}")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(tae.parameters(), grad_clip)
            optimizer.step()

            batch_losses.append(loss.item())

        clust_assign = new_clust_assign.clone()
        tae.centers = (tae.update_centers(Y, clust_assign) + epoch * tae.centers) / (epoch + 1)

        warmup_losses.append(np.mean(batch_losses))
        pbar_warmup.update(warmup_losses[-1])

    
    clust_wise_data = tae._collect_data(X, Y, clust_assign)
    dataloaders = [(c, GenericDataset(x, y).get_dataloader(batch_size=batch_size, shuffle=True)) for (c,x,y) in clust_wise_data]    

    batched_epochs = (epochs - warmup_epochs)
    
    if verbose: print(f"PHASE 2: Batched — {batched_epochs}/{epochs}")
    clust_losses = []
    for data in dataloaders:
        ae = tae.autoencoders.AE[data[0]]
        optimizer = torch.optim.Adam(ae.parameters(), lr)
        ae, losses = train_ae(ae, data[1], optimizer, epochs=batched_epochs, verbose=verbose)
        clust_losses.append(losses)

    return warmup_losses, clust_losses, clust_assign.argmax(dim=0)