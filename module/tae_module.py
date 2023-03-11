import torch
import torch.nn as nn
import math
from sklearn.cluster import kmeans_plusplus
from train_utils import ProgressBar, EarlyStopping
import copy

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, bias, activation=None):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias)
        self.act = activation

    def forward(self, x):
        if self.act is None:
            return self.layer(x)
        
        return self.act(self.layer(x))

class Autoencoder(nn.Module):
    def __init__(self, enc_channels, dec_channels, bias=True, enc_activations=nn.ReLU(), dec_activations=nn.ReLU()):
        super().__init__()
        self.encoder = nn.Sequential()

        if enc_channels[-1] != dec_channels[0]: 
            print("[WARN] First shape of dec_channels does not match the terminal channel in enc_channels, proceeding with additional layer...")
            dec_channels = (enc_channels[-1],)+dec_channels

        for i in range(len(enc_channels)-1):
            self.encoder.add_module(f'enc_dense{i}', DenseBlock(enc_channels[i], enc_channels[i+1], bias=bias, activation=enc_activations))

        self.decoder = nn.Sequential()
        for i in range(len(dec_channels)-1):
            self.decoder.add_module(f'dec_dense{i}', DenseBlock(dec_channels[i], dec_channels[i+1], bias=bias, activation=dec_activations))

    def forward(self, x, return_embed=False):
        x = x.float()
        embed = self.encoder(x)
        out = self.decoder(embed)

        if return_embed: return embed, out
        return out

class GroupedAE(nn.Module):
    def __init__(self, autoencoder, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.AE = nn.ModuleList(copy.deepcopy(autoencoder) for i in range(n_clusters))

    def forward(self, x, centers, return_embed=False):
        embed, out = self.AE[0](x - centers[0].reshape(x.shape[1:]), True)
        embed, out = embed.reshape((len(x),1)+embed.shape[1:]), out.reshape((len(x),1)+out.shape[1:])

        for n in range(1,self.n_clusters):
            e, o = self.AE[n](x - centers[n].reshape(x.shape[1:]), True)
            embed = torch.cat((embed, e.reshape((len(x),1)+e.shape[1:])), dim=1)
            out = torch.cat((out, o.reshape((len(x),1)+o.shape[1:])), dim=1)

        if return_embed: return embed, out         # (batch, n_clusters, 3, 32, 32)
        return out

class TensorizedAutoencoder(nn.Module):
    def __init__(self, autoencoder, Y_data, n_clusters, regularizer_coef=0.01):
        super().__init__()
        self.n_clusters = n_clusters
        self.centers = None
        
        self.autoencoders = GroupedAE(autoencoder, n_clusters)
        self.mse = nn.functional.mse_loss
        self.reg = regularizer_coef

        self.random_state = None    # set manually if needed, call model._init_clusters(Y) again
        self._init_clusters(Y_data)

    def _init_clusters(self, Y):
        self.centers = kmeans_plusplus(Y.reshape(len(Y), -1).numpy(), n_clusters=self.n_clusters, random_state=self.random_state)

    def forward_pass(self, x, clust_idx=None, centers=None, return_embed=False):
        """
        Getting output for a given batch of inputs for 
        1. computed centers and Nearest-Center-Assignment (NCA)
        2. input override centers and NCA
        3. input override assigned clusters
        """
        if self.centers is None and centers is None:
            print("Cannot compute output!") 
            return

        device = x.device
        centers = centers if centers is not None else self.centers
        centers = centers.to(device).reshape(self.n_clusters,-1)
        if clust_idx is None:
            clust_idx = self._assign_centers_to_data(x, centers)

        embed, out = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for idx in range(len(x)):
            samp = x[idx:idx+1]     # 1, 3, 32, 32
            clust = clust_idx[idx]
            e, o = self.autoencoders.AE[clust](samp-centers[clust].reshape(samp.shape), True)
            embed = torch.cat((embed, e))
            out = torch.cat((out, o))

        if return_embed: return embed, out
        return out
    
    def compute_loss(self, x, y, centers, assigned_clusters):
        x_ = x.clone()  # [batch, 3, 32, 32]
        y_ = y.clone()  # [batch, 3, 32, 32]
        
        embed, out = self.autoencoders(x_, centers, True)   # [batch, n_clusters, 3, 32, 32]

        losses, new_idxs = torch.tensor([]).to(x_.device), torch.tensor([])
        for idx in range(len(x_)):
            embed_, out_, true_ = embed[idx], out[idx], y_[idx]

            assert out_.shape == [self.n_clusters] + x_.shape[1:]
            # embed_, out_: (n_clusters, 3, 32, 32)

            embed_norm = embed_.clone()
            while len(embed_norm.shape) > 1:
                embed_norm = torch.norm(embed_norm, dim=1)

            dims = [i for i in range(1,len(x_.shape[1:]))]
            mse_proxy = torch.sum((out_ - true_.reshape((1)+true_.shape)) ** 2, dim=dims)
            loss_proxy = mse_proxy + self.reg * (embed_norm ** 2)

            new_idx = loss_proxy.argmin()   #reassigned clust
            
            clust = assigned_clusters[idx]  
            loss = self.mse(embed_[clust], true_) + self.reg * (embed_norm[clust] ** 2) #using currently assigned

            losses = torch.cat((losses, loss))
            new_idxs = torch.cat((new_idxs, new_idx))

        new_indices = nn.functional.one_hot(new_idxs).T
        losses = sum(losses)/len(losses)        

        return losses, new_indices

    def _update_centers(self, Y, clust_assign):
        clust_assign.to(Y.device)
        new_centers = clust_assign.float() @ Y.reshape(len(Y), -1).float()
        new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ \
                        torch.ones(1, self.centers.shape[1], dtype=torch.float)
        new_centers = new_centers / new_norm
        return new_centers

    def _assign_centers_to_data(self, data, centers=None):
        centers = centers or self.centers
        centers = centers.reshape(self.n_clusters, -1)

        assignments = torch.tensor([])
        for i in range(self.n_clusters):
            d = torch.norm(data.reshape(len(data), -1) - self.centers[i], dim=1).reshape(-1,1)
            assignments = torch.cat((assignments,d), dim=1)
        return assignments.argmin(dim=1)

    def train(self, X, Y, Dataset, epochs, lr, batch_size, **dataset_kwargs):
        self.configure_optimizers(lr)

        device = self.device
        X = X.to(device)
        Y = Y.to(device)

        clust_assign = self._assign_centers_to_data(Y)
        clust_assign_onehot = nn.functional.one_hot(clust_assign).T.float()

        self.centers = self._update_centers(Y, clust_assign)

        dataset = Dataset(X, Y, **dataset_kwargs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

        pbar = ProgressBar(epochs)
        es = EarlyStopping()

        losses = []
        for epoch in pbar.bar:
            
            for batch_idx, (x, y) in enumerate(dataloader):
                index = batch_idx * batch_size
                x, y = x.to(device), y.to(device)
                loss, new_indices = self.compute_loss(x, y, self.centers, clust_assign)

                new_indices = torch.squeeze(new_indices)
                new_indices = nn.functional.one_hot(new_indices).T.float()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                clust_assign_onehot[:,index:index+batch_size] = new_indices
                clust_assign = clust_assign_onehot.argmax(dim=1)

                new_centers = self._update_centers(Y, clust_assign) 
                self.centers = (index * self.centers + batch_size * new_centers) / (index + batch_size)

                losses.append(loss.item)

            es.update(losses[-1])
            pbar.update(epoch, losses[-1], es.es)
        
        return losses
    
    def configure_optimizers(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)