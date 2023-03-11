import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Autoencoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc = nn.Linear(in_features=in_feature, out_features=embed, bias=False).to(device)
        # decoder
        self.dec = nn.Linear(in_features=embed, out_features=in_feature, bias=False).to(device)
        self.linear = linear
        # self.double()

    def forward(self, x):
        x = self.enc(x).to(device)
        if self.linear == False:
            x = F.relu(x).to(device)
        x = self.dec(x).to(device)
        return x
    
class VAE(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(VAE, self).__init__()
        # encoder
        self.encmu = nn.Linear(in_features=in_feature, out_features=embed, bias=False).to(device)
        self.encsig = nn.Linear(in_features=in_feature, out_features=embed, bias=False).to(device)
        # decoder
        self.dec = nn.Linear(in_features=embed, out_features=in_feature, bias=False).to(device)
        self.linear = linear
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

        # self.double()

    def forward(self, x, return_embed=False):
        mu = self.encmu(x)
        sigma = torch.exp(self.encsig(x))
        z = self.N.sample(mu.shape) * sigma + mu
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma ** 2) - 1).sum() / 2
        
        if return_embed: return z, self.dec(z)
        return self.dec(z)


class TensorisedAEloss(nn.Module):
    def __init__(self, in_feature, embed, reg, num_clusters=2, linear=True, CNN=False):
        super(TensorisedAEloss, self).__init__()

        self.AE = nn.ModuleList()
        # add num_clusters AE
        self.n_clust = num_clusters
        for i in range(num_clusters):
            # note: this should be written better so that the network is passed
            if CNN:
                self.AE.append(CNN_Autoencoder(embed))
            else:
                self.AE.append(Autoencoder(in_feature, embed, linear))

        self.mse = nn.MSELoss()
        self.reg = reg
        self.CNN = CNN

    def forward(self, X, centers, i, clust_assign, X_out=None):
        if X_out == None:
            X_out = X

        loss = 0
        loss_clust_idx = -1
        loss_clust = torch.inf

        for j in range(self.n_clust):
            x = X - centers[j]
            x_out = X_out - centers[j]

            if self.CNN:
                # this can def be optimized
                x = x.reshape(28, 28)
                x = torch.unsqueeze(x, dim=0)
                x = torch.unsqueeze(x, dim=0)

                x_out = x_out.reshape(28, 28)
                x_out = torch.unsqueeze(x_out, dim=0)
                x_out = torch.unsqueeze(x_out, dim=0)

                l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j].enc1(x))))
            else:
                l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j].enc1(x))))

            if loss_clust > l:
                loss_clust = l
                loss_clust_idx = j
            l = clust_assign[j][i] * l

            loss += l
        return loss, loss_clust_idx

class TensorisedAElossMod(nn.Module):
    def __init__(self, in_feature, embed, reg, num_clusters=2, linear=True):
        super(TensorisedAElossMod, self).__init__()

        self.AE = nn.ModuleList()
        # add num_clusters AE
        self.n_clust = num_clusters
        for i in range(num_clusters):
            # note: this should be written better so that the network is passed
            self.AE.append(VAE(in_feature, embed, linear))

        self.mse = nn.MSELoss()
        self.reg = reg

    def forward(self, X, centers, i, clust_assign, X_out=None, use_all=1):
        if X_out == None:
            X_out = X

        loss = 0
        loss_clust_idx = -1
        loss_clust = torch.inf

        clusts = range(self.n_clust) if use_all else [clust_assign[i].argmax()]

        for j in clusts:
            x = X - centers[j]
            x_out = X_out - centers[j]

            l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j](x, return_embed=True)[0]))) + self.AE[j].kl

            if loss_clust > l:
                loss_clust = l
                loss_clust_idx = j
            l = clust_assign[j][i] * l

            loss += l
        return loss, loss_clust_idx


class CNN_Autoencoder(nn.Module):
    def __init__(self,embed):
        super(CNN_Autoencoder,self).__init__()
        self.enc = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, embed),
            # 10
            # nn.Softmax(),
            )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(embed, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
    def forward(self, x):
        enc = self.enc(x)
        dec = self.decoder(enc)
        return dec