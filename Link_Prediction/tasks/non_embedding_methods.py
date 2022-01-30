import sys

from torchaudio import datasets
sys.path.insert(0, ".")

from pathlib import Path
import re
import pandas as pd 
import numpy as np
import networkx as nx
from utils.load_numpy_dataset import load_numpy_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def create_non_embedding_data(path):
    
    dir_graph = ['soc-Epinions','Wiki-Vote']
    undir_graph = ['CA-AstroPh','CA-HepTh','ppi']

    dataset_name = re.sub('\([\w]*\)','',str(path).split('/')[-1])

    index, train, train_neg, test, test_neg = load_numpy_data(path)
    
    # creating train dataset
    print('creating train dataset for {} ...'.format(dataset_name))
    train_df = pd.DataFrame(train, columns=['Source', 'Target'])
    train_df['Connection'] = 1
    train_neg_df = pd.DataFrame(train_neg, columns=['Source', 'Target'])
    train_neg_df['Connection'] = 0

    frames = [train_df, train_neg_df]
    train_edges = pd.concat(frames)
    train_edges = train_edges.sort_values(['Source', 'Target'], inplace=False)
    train_edges = train_edges[train_edges['Source'] != train_edges['Target']]
    print(f'train edges shape for {dataset_name}: {train_edges.shape}')

    train_edges['Edges'] = list(zip(train_edges.Source, train_edges.Target))
    train_edges = train_edges.drop(columns=train_edges.columns[:2], inplace=False)
    train_edges = train_edges.set_index('Edges')
    train_edges.to_csv(f'datasets_pp/original/{dataset_name}/{dataset_name}_train_edges.csv')
    print('train edges saved to csv')


    #creating test dataset
    print('creating test dataset for {} ...'.format(dataset_name))
    test_df = pd.DataFrame(test, columns=['Source', 'Target'])
    test_df['Connection'] = 1
    test_neg_df = pd.DataFrame(test_neg, columns=['Source', 'Target'])
    test_neg_df['Connection'] = 0

    frames = [test_df, test_neg_df]
    test_edges = pd.concat(frames)
    test_edges = test_edges.sort_values(['Source', 'Target'], inplace=False)
    test_edges = test_edges[test_edges['Source'] != test_edges['Target']]
    print(f'test edges shape for {dataset_name}: {test_edges.shape}')
    
    test_edges['Edges'] = list(zip(test_edges.Source, test_edges.Target))
    test_edges = test_edges.drop(columns=test_edges.columns[:2], inplace=False)
    test_edges = test_edges.set_index('Edges')
    test_edges.to_csv(f'datasets_pp/original/{dataset_name}/{dataset_name}_test_edges.csv')
    print('test edges saved to csv')


    # create new graph
    frames = [train_df, test_df]
    edges = pd.concat(frames)
    edges = edges.sort_values(['Source', 'Target'], inplace=False)
    edges = edges[edges['Source'] != edges['Target']]
    edges = edges.drop(columns=['Connection'], inplace=False)
    edges = edges.reset_index(drop=True)
    edges = edges[['Source', 'Target']]
    edges.to_csv(f'datasets_pp/original/{dataset_name}/{dataset_name}_edges.csv')
    
    if dataset_name in dir_graph:
        G = nx.from_pandas_edgelist(edges, 'Source', 'Target', create_using=nx.DiGraph())
        nx.write_gexf(G, f"graphs_pp/original/{dataset_name}.gexf")
        print(f'{dataset_name} graph saved as gexf')

    elif dataset_name in undir_graph:
        G = nx.from_pandas_edgelist(edges, 'Source', 'Target', create_using=nx.Graph())
        nx.write_gexf(G, f"graphs_pp/original/{dataset_name}.gexf")
        print(f'{dataset_name} graph saved as gexf')

    else:
        print('Invalid Dataset')
    
    return(G, train_edges, test_edges, dataset_name)


def add_non_embedding_features(G,dataset,dataset_name):

    print(dataset.head())

    if G.__class__.__name__ == 'DiGraph':
        G = nx.to_undirected(G)
        print('graph converted to undirected')
    
    # Common Neighbors
    print('adding common neighbors feature ...')
    dataset['Common_Neigh'] = [len(list(nx.common_neighbors(G, e[0],e[1]))) for e in dataset.index]
    
    # Jaccard Coefficient
    print('adding jaccard coefficient feature ...')
    temp = pd.DataFrame()
    jaccard = list(nx.jaccard_coefficient(G, list(dataset.index)))
    temp['Jaccard_Coef'] = [i[2] for i in jaccard]
    temp['index'] = [(i[0],i[1]) for i in jaccard]
    temp.set_index('index', inplace=True)
    dataset = dataset.join(temp, how='inner')

    # Adamic-Adar Index
    print('adding adamic-adar index feature ...')
    temp = pd.DataFrame()
    adamic_adar = list(nx.adamic_adar_index(G, list(dataset.index)))
    temp['Adamic_Adar'] = [i[2] for i in adamic_adar]
    temp['index'] = [(i[0],i[1]) for i in adamic_adar]
    temp.set_index('index', inplace=True)
    dataset = dataset.join(temp, how='inner')
   
    # scaling features
    print('scaling features ...')
    scaler = MinMaxScaler()
    dataset_np = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset_np, index=dataset.index ,columns=['Connection', 'Common_Neigh', 'Jaccard_Coef', 'Adamic_Adar'])

    # saving dataset
    dataset.to_csv(f'datasets_pp/{dataset_name}/{dataset_name}_non_embedding_features.csv')

    print(dataset.head())

    return(dataset,dataset_name)


def plot_feature_correlation(dataset, dataset_name, train):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap = True)
    _ = sns.heatmap(dataset.corr(), cmap = colormap, square=True, cbar_kws={'shrink':.9 }, ax=ax, annot=True,
                                                        linewidths=0.1,vmax=1.0, linecolor='black', annot_kws={'fontsize':12 })
    
    plt.title(f'Pearson Correlation of Features for {dataset_name}', y=1.05, size=15)

    if train==True:
        plt.savefig(f'plots/feature_correlation_train_{dataset_name}.png')
        plt.show()
    else:
        plt.savefig(f'plots/feature_correlation_test_{dataset_name}.png')
        plt.show()



if __name__ == "__main__":
    folders = Path('datasets_pp/original/').glob('*')
    for i in folders:
        print(str(i))
        G, train_edges, test_edges, dataset_name = create_non_embedding_data(str(i))

        train_features, train_features_dataset_name = add_non_embedding_features(G, train_edges, dataset_name)
        plot_feature_correlation(train_features, train_features_dataset_name, train=True)
        
        test_features, test_features_dataset_name = add_non_embedding_features(G, test_edges, dataset_name)
        plot_feature_correlation(test_features, test_features_dataset_name, train=False)