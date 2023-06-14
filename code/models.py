import torch
import torch.nn as nn

from functools import partial


class Block(nn.Module):
    def __init__(self, layer_type: nn.Module, in_features, out_features, activation=None, **layer_kwargs):
        super().__init__()
        self.layer = layer_type(in_features, out_features, **layer_kwargs)
        self.act = activation

    def forward(self, x):
        if self.act is None:
            return self.layer(x)

        return self.act(self.layer(x))


class BaseAutoencoder(nn.Module):
    def __init__(self, layer_type: nn.Module, enc_channels, dec_channels, bias=True, activations=nn.ReLU(), device=None,
                 **layer_kwargs):
        """
        REQUIRES: device, enc_channels, dec_channels, activations (nn.ReLU(), nn.GELU(), etc)
        """
        super().__init__()

        self.enc_channels, self.dec_channels = enc_channels, dec_channels
        self.bias = bias
        self.activations = activations
        self.device = device or 'cpu'

        if self.enc_channels[-1] != self.dec_channels[0]:
            print("[WARN] First shape of dec_channels does not match the terminal channel in enc_channels, proceeding "
                  "with additional layer...")
            self.dec_channels = (self.enc_channels[-1],) + self.dec_channels

        self.enc = nn.Sequential()
        for i in range(len(self.enc_channels) - 1):
            self.enc.add_module(f'enc_dense{i}',
                                Block(layer_type, self.enc_channels[i], self.enc_channels[i + 1], bias=self.bias,
                                      activation=self.activations, **layer_kwargs).to(self.device))

        self.dec = nn.Sequential()
        for i in range(len(self.dec_channels) - 1):
            self.dec.add_module(f'dec_dense{i}',
                                Block(layer_type, self.dec_channels[i], self.dec_channels[i + 1], bias=self.bias,
                                      activation=self.activations, **layer_kwargs).to(self.device))

    def forward(self, x, return_embed=False):
        x = x.to(self.device).float()
        embed = self.enc(x)
        out = self.dec(embed)

        if return_embed: return embed, out
        return out

    @classmethod
    def create_AE(cls, layer_type):
        return partial(cls, layer_type=layer_type)


class GroupedModel(nn.Module):
    def __init__(self, n_clusters, model_class, **kwargs):
        super().__init__()
        self.n_clusters = n_clusters

        self.AE = nn.ModuleList(model_class(**kwargs) for i in range(n_clusters))
        self.device = 'cpu' if kwargs.get('device') is None else kwargs['device']

    def _return_embed(self, embed, out, flag):
        return (embed, out) if flag else out

    def forward_with_clust(self, x, centers, clust, return_embed=False):
        """ 
        To be used in the batched phase, computes output for input x, all belonging to the same cluster 
        """
        x, centers = x.to(self.device), centers.to(self.device)
        embed, out = self.AE[clust](x - centers[clust], True)

        return self._return_embed(embed, out, return_embed)

    def forward_with_centers(self, x, centers, return_embed=False):
        """ 
        To be used in warmup phase, computes output for input x for all clusters. 
        Output format (batch, n_clusters, <data shape>)
        """
        x, centers = x.to(self.device), centers.to(self.device)
        embed, out = self.AE[0](x - centers[0].reshape(x.shape[1:]), True)
        embed, out = embed.reshape((len(x), 1) + embed.shape[1:]), out.reshape((len(x), 1) + out.shape[1:])

        for n in range(1, self.n_clusters):
            e, o = self.AE[n](x - centers[n].reshape(x.shape[1:]), True)
            embed = torch.cat((embed, e.reshape((len(x), 1) + e.shape[1:])), dim=1)
            out = torch.cat((out, o.reshape((len(x), 1) + o.shape[1:])), dim=1)

        return self._return_embed(embed, out, return_embed)

    def forward(self, *args):
        raise NotImplementedError("Use one of `forward_with_centers` or `forward_with_clust`")
