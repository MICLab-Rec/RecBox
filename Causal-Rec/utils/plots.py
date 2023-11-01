import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns


def plot_low_embedding(embedding,type = 'tsne'):
    '''

    @param embedding:
    @param type: the dimensionality reduction method , tsne or pca, default is tsne
    @return:
    '''

    if type == 'tsne':
        X = TSNE(n_components=2,random_state=33,perplexity= min(50,embedding.shape[0] - 1)).fit_transform(embedding)
    elif type == 'tsne':
        X = PCA(n_components=2).fit_transform(embedding)
    else:
        raise NotImplementedError('the dimensionality reduction method no exist !!!!!!')
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("bright", embedding.shape[0])
    sns.scatterplot(x= X[:,0],y= X[:,1], hue=range(embedding.shape[0]), legend='full', palette=palette)
    # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()

def get_graph_from_adj(adj:np.array):
    adj[np.abs(adj) < 0.1] = 0
    G = nx.from_numpy_matrix(adj,create_using=nx.DiGraph)
    nx.draw_networkx(G,pos=nx.circular_layout(G),with_labels=True,alpha=0.5,node_color='yellow',node_shape='s',
                 linewidths=4,
                 width=2,edge_color='blue',style='--',
                 font_size=15,font_color='blue')
    plt.show()


def plot_hotmap(data):

    data[np.abs(data) < 0.1] = 0
    sns.heatmap(data,annot = True, fmt = ".2f", linewidths = 0.3, linecolor = "grey", cmap = "RdBu_r")
    plt.show()


def plot_line(x,y,label):
    sns.scatterplot(x=x, y=y, label=label)
    plt.show()
