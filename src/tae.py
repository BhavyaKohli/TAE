import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import kmeans_plusplus


def to_npy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def mse(x, y, dim=None):
    return F.mse_loss(x, y, reduction='none').mean(dim=dim) if dim is not None else F.mse_loss(x, y)


class TensorizedAutoencoder(nn.Module):
    def __init__(self, grouped_model, Y_data, regularizer_coef=0.01):
        super(TensorizedAutoencoder, self).__init__()
        self.n_clusters = grouped_model.n_clusters
        self.centers = None
        self.device = grouped_model.device

        self.autoencoders = grouped_model
        self.reg = regularizer_coef

        self.random_state = None  # set manually if needed, call model._init_clusters(Y) again
        self.data_shape = tuple(Y_data.shape[1:])
        self._init_clusters(to_npy(Y_data))

    def _init_clusters(self, Y):
        centers = kmeans_plusplus(Y.reshape(len(Y), -1), n_clusters=self.n_clusters, random_state=self.random_state)[0]
        self.centers = torch.from_numpy(centers).reshape((self.n_clusters,) + self.data_shape).to(self.device)
        assert self.centers.shape == (self.n_clusters,) + self.data_shape

    def _get_flat_centers(self):
        return self.centers.reshape(self.n_clusters, -1)

    def forward_with_clust(self, x, clust, return_embed=False):
        return self.autoencoders.forward_with_clust(x, self.centers, clust, return_embed)

    def forward_with_centers(self, x, return_embed=False):
        return self.autoencoders.forward_with_centers(x, self.centers, return_embed)

    def assign_centers_to_data(self, data, one_hot=False, centers=None):
        centers = centers or self.centers
        centers = centers.reshape(self.n_clusters, -1)
        data, centers = data.to(self.device), centers.to(self.device)

        assignments = torch.tensor([], device=self.device)
        for i in range(self.n_clusters):
            d = torch.norm(data.reshape(len(data), -1) - centers[i], dim=1).reshape(-1, 1)
            assignments = torch.cat((assignments, d), dim=1)
        if one_hot: return F.one_hot(assignments.argmin(dim=1), self.n_clusters).T
        return assignments.argmin(dim=1)

    def update_centers(self, Y, clust_assign):
        assert clust_assign.shape == (self.n_clusters, len(Y)), "check if clust_assign is in one-hot format"
        Y, clust_assign = Y.to(self.device), clust_assign.to(self.device)

        new_centers = clust_assign.float() @ Y.reshape(len(Y), -1).float()
        new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1).to(self.device) @ \
                   torch.ones(1, self._get_flat_centers().shape[1], dtype=torch.float).to(self.device)
        new_centers = new_centers / new_norm
        return new_centers

    def compute_loss_warmup(self, x, y, clust_assign):
        x, y, clust_assign = x.to(self.device), y.to(self.device), clust_assign.to(self.device)

        clusts = clust_assign.argmax(dim=0)
        b = x.shape[0]

        x_, y_ = x.clone(), y.clone()

        embed, out = self.autoencoders.forward_with_centers(x_, self.centers, True)
        # (batch, n_clusters, <data shape>)

        new_assignment = torch.zeros((self.n_clusters, b), dtype=int)
        loss = 0
        for samp in range(b):
            true, e, o = y_[samp][None, :].repeat_interleave(self.n_clusters, dim=0), embed[samp], out[samp]

            e_, o_ = e.reshape(self.n_clusters, -1), o.reshape(self.n_clusters, -1)
            true_ = true.reshape(len(true), -1)

            loss_proxy = mse(o_, true_, dim=1) + self.reg * (torch.norm(e_, dim=1) ** 2)
            # shapes(loss_proxy)

            new_center = loss_proxy.argmin(dim=0)
            loss += mse(o_[clusts[samp]], true_[0]) + self.reg * (torch.norm(e_) ** 2)

            new_assignment[:, samp][new_center] = 1

        return loss, new_assignment

    @staticmethod
    def collect_data(x, y, clust_assign):
        clusts = clust_assign.argmax(dim=0)
        return [(i, x[clusts == i], y[clusts == i]) for i in torch.unique(clusts)]

    def compute_loss_batch(self, x, y, clust_assign):
        collected = self.collect_data(x, y, clust_assign)
        loss = 0
        for c, x, y in collected:
            x_ = x.clone() - self.centers[c]
            y_ = y.clone() - self.centers[c]

            embed, out = self.forward_with_clust(x_.squeeze(), c, return_embed=True)
            loss += mse(out, y_) + self.reg * (torch.norm(embed) ** 2)

        return loss / len(collected)
