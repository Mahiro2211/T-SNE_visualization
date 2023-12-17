import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn import manifold
from scipy.spatial import distance


def plot_distribution(data, path):
    _, D = data.shape
    plt.figure(figsize=(32, 32));
    for i in range(1, 32 + 1):
        plt.subplot(D // 4, 4, i);
        commutes = pd.Series(data[:, i - 1])
        commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');
        plt.title(f'{i}bit')

    plt.savefig(f"{path}/data_distribution.png")


def plot_distance(db_feats, db_label, query_feats, query_label, path):
    S = np.matmul(db_label, query_label.transpose())
    N = np.sum(S == 1)

    plt.figure(figsize=[16, 6])
    plt.subplot(121)
    cosine_32bit = distance.cdist(db_feats, query_feats, metric='cosine') / 2
    plt.title('cosine distribution')
    commutes = pd.Series(np.hstack((np.random.choice(cosine_32bit[S == 1].flatten(), N), \
                                    np.random.choice(cosine_32bit[S == 0].flatten(), N))))
    commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');
    plt.subplot(122)
    euclidean_32bit = distance.cdist(db_feats, query_feats, metric='euclidean')
    plt.title('euclidean distribution')
    commutes = pd.Series(np.hstack((np.random.choice(euclidean_32bit[S == 1].flatten(), N), \
                                    np.random.choice(euclidean_32bit[S == 0].flatten(), N))))
    commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');
    plt.savefig(f"{path}/distance_distribution.png")

    plt.figure(figsize=[16, 6])
    plt.subplot(121)
    plt.title('cosine similar distribution')
    commutes = pd.Series(cosine_32bit[S == 1].flatten())
    commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');
    plt.subplot(122)
    plt.title('cosine dissimilar distribution')
    commutes = pd.Series(cosine_32bit[S == 0].flatten())
    commutes.plot.hist(grid=True, bins=200, rwidth=0.9, color='#607c8e');
    plt.savefig(f"{path}/similarity_distribution.png")


def plot_tsne(epoch,data, label, path, R=2000):
    if label.ndim > 1:
        label = label.argmax(axis=1)
    # colors = np.random.rand(30)
    plt.figure(figsize=(16, 12));
    embed = TSNE(n_components=2, perplexity=30, lr=50, eps=1e-9, n_iter=1000000, device='cuda').fit_transform(data[:R])
    plt.scatter(embed[:, 0], embed[:, 1], c=label[:R], s=10)
    plt.savefig(f"{path}/data_t-SNE+f'{epoch}'.png")


class TSNE(object):

    def __init__(self, n_components=2, perplexity=30, lr=1, eps=1e-9, n_iter=2000, device='cpu'):
        self.perplexity = perplexity
        self.lr = lr
        self.eps = eps
        self.n_iter = n_iter
        self.device = device
        self.n_components = n_components

    def t_distribution(self, y):
        n = y.shape[0]
        dist = torch.sum((y.reshape(n, 1, -1) - y.reshape(1, n, -1)) ** 2, -1)
        affinity = 1 / (1 + dist)
        affinity *= (1 - torch.eye(n, device=self.device))  # set diag to zero
        q = affinity / affinity.sum() + self.eps
        return q

    def fit_transform(self, x):
        dist2 = distance.squareform(distance.pdist(x, metric='sqeuclidean'))
        p = distance.squareform(manifold._t_sne._joint_probabilities(dist2, self.perplexity, False)) + self.eps

        p = torch.tensor(p, device=self.device, dtype=torch.float32).reshape(-1)
        log_p = torch.log(p)

        y = torch.randn([dist2.shape[0], self.n_components], device=self.device, requires_grad=True)
        optimizer = optim.Adam([y], lr=self.lr)
        criterion = torch.nn.KLDivLoss()

        for i_iter in range(self.n_iter):
            q = self.t_distribution(y).reshape(-1)
            loss = (p * (log_p - torch.log(q))).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return y.detach().cpu().numpy()

import scipy.io as scio

'''
    introduction
    embedding : shape [Batch , embed_size]
    label : shape [Batch , ]
'''

import glob
import numpy as np

files = glob.glob('./1/*.mat')
'''
    make your embedding a independent .mat file like [1.mat , 2.mat ....]
'''
tot = 0
tot_code = 0
for i , file in enumerate(files):
    # automatic set a global variabel for your 
    data = scio.loadmat(file)
    code = data['hash']
    code_len = code.shape[0]
    print(f'{i} code_len {code_len}')
    name = 'the' + str(i) + 'code'
    label_name = 'the' + str(i) + 'label'
    tot_code += code_len
    globals()[name] = code
    if i == 0:
        globals()[label_name] = np.zeros((code_len))
        print(f'label {i} {globals()[label_name].shape}')
        tot += globals()[label_name].shape[0]
        print(globals()[label_name])
    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        globals()[label_name] = np.ones((code_len,))
        globals()[label_name][globals()[label_name] == 1] = i
        print(f'label {i} {globals()[label_name].shape}')
        # print(globals()[label_name])
        tot += globals()[label_name].shape[0]
    

print(tot == tot_code)

for i in range(len(files)):
    #concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
    if i == 0:
        codes = globals()[f'the' + str(i) + 'code']
        labels = globals()[f'the' + str(i) + 'label']
    else:
        codes = np.concatenate((codes , globals()[f'the' + str(i) + 'code']))
        labels = np.concatenate((labels , globals()[f'the' + str(i) + 'label']))
print(f'code length is {codes.shape}')
print(f'label length is {labels.shape}')
# print(label)

plot_tsne(epoch=0 , data=codes,label=labels , R=codes.shape[0] , path='.')




    