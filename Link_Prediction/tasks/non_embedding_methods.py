import sys
sys.path.insert(0, ".")

from pathlib import Path
import re
import pandas as pd 
import numpy as np
import networkx as nx
from utils.load_numpy_dataset import load_numpy_data
from tqdm import tqdm
from math import log
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


def cn_dir(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def jc_dir(g, i, j):
    un = len(set(g[i])|set(g[j]))
    intrsc = len(cn_dir(g, i, j))
    try:
        jc = intrsc/un
    except:
        jc = 0
    return(jc)

def aa_dir(g, i, j):
    try:
        aa = sum(1/log(len(set(g[w]))) for w in cn_dir(g, i, j))
    except:
        aa = 0
    return(aa)

def cn_undir(g, i, j):
    return set(g.neighbors(i)).intersection(set(g.neighbors(j)))

def jc_undir(g, i, j):
    un = len(set(g[i])|set(g[j]))
    intrsc = len(cn_undir(g, i, j))
    try:
        jc = intrsc/un
    except:
        jc = 0
    return(jc)

def aa_undir(g, i, j):
    try:
        aa = sum(1/log(len(set(g[w]))) for w in cn_undir(g, i, j))
    except:
        aa = 0
    return(aa)


def create_non_embedding_data(path):
    
    dir_graph = ['soc-epinions','wiki-vote']
    undir_graph = ['ca-astroph','ca-hepth','ppi']

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

    if dataset_name.lower() in dir_graph:
        G_train = nx.from_pandas_edgelist(train_edges, 'Source', 'Target', create_using=nx.DiGraph())

    elif dataset_name.lower() in undir_graph:
        G_train = nx.from_pandas_edgelist(train_edges, 'Source', 'Target', create_using=nx.Graph())

    else:
        raise ValueError('Invalid Graph')

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

    if dataset_name.lower() in dir_graph:
        G_test = nx.from_pandas_edgelist(test_edges, 'Source', 'Target', create_using=nx.DiGraph())

    elif dataset_name.lower() in undir_graph:
        G_test = nx.from_pandas_edgelist(test_edges, 'Source', 'Target', create_using=nx.Graph())

    else:
        raise ValueError('Invalid Graph')
    
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
    
    if dataset_name.lower() in dir_graph:
        G_original = nx.from_pandas_edgelist(edges, 'Source', 'Target', create_using=nx.DiGraph())
        nx.write_gexf(G_original, f"graphs_pp/original/{dataset_name}.gexf")
        print(f'{dataset_name} graph saved as gexf')

    elif dataset_name.lower() in undir_graph:
        G_original = nx.from_pandas_edgelist(edges, 'Source', 'Target', create_using=nx.Graph())
        nx.write_gexf(G_original, f"graphs_pp/original/{dataset_name}.gexf")
        print(f'{dataset_name} graph saved as gexf')

    else:
        raise ValueError('dataset name not found')
    
    return(G_test, G_train, train_edges, test_edges, dataset_name)


def add_non_embedding_features(G,dataset,dataset_name,train):

    print(dataset.head())
    
    if G.__class__.__name__ == 'DiGraph':
        print(f'{dataset_name} graph is directed')
        edges = dataset.index.tolist()
        cn, jc, aa = [], [], []
        for e in tqdm(edges):
            cn.append(len(cn_dir(G, e[0], e[1])))
            jc.append(jc_dir(G, e[0], e[1]))
            aa.append(aa_dir(G, e[0], e[1]))

        dataset['CN'] = cn
        dataset['JC'] = jc
        dataset['AA'] = aa

    
    elif G.__class__.__name__ == 'Graph':
        print(f'{dataset_name} graph is undirected')
        edges = dataset.index.tolist()
        cn, jc, aa = [], [], []
        for e in tqdm(edges):
            cn.append(len(cn_undir(G, e[0], e[1])))
            jc.append(jc_undir(G, e[0], e[1]))
            aa.append(aa_undir(G, e[0], e[1]))

        dataset['CN'] = cn
        dataset['JC'] = jc
        dataset['AA'] = aa

    else:
        raise ValueError('Invalid Graph')
   
    # scaling features
    print('scaling features ...')
    scaler = MinMaxScaler()
    dataset_np = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset_np, index=dataset.index ,columns=['Connection', 'CN', 'JC', 'AA'])

    # saving dataset
    if train==True:
        dataset.to_csv(f'datasets_pp/original/{dataset_name}/{dataset_name}_train_non_embedding_features.csv')
    else:
        dataset.to_csv(f'datasets_pp/original/{dataset_name}/{dataset_name}_test_non_embedding_features.csv')

    print(dataset.head())

    return(dataset,dataset_name)


def calculate_roc_auc(dataset,dataset_name):
    
    ytrue = dataset['Connection'].to_numpy()
    ytrue = ytrue.astype(float)
    
    yscore_jc = dataset['JC'].to_numpy()
    yscore_jc = yscore_jc.astype(float)
    
    yscore_aa = dataset['AA'].to_numpy()
    yscore_aa = yscore_aa.astype(float)
    
    yscore_cn = dataset['CN'].to_numpy()
    yscore_cn = yscore_cn.astype(float)
    
    rc_jc = roc_auc_score(y_true=ytrue, y_score=yscore_jc,  average='micro')
    rc_aa = roc_auc_score(y_true=ytrue, y_score=yscore_aa,  average='micro')
    rc_cn = roc_auc_score(y_true=ytrue, y_score=yscore_cn,  average='micro')
    
    print(f"Dataset: {dataset_name}")
    print(f"ROC-score:\n Jacard Coefficent:{rc_jc:1.3f}, Admic Adar:{rc_aa:1.3f}, Common neighbours:{rc_cn:1.3f}")
    
    return rc_jc,rc_aa,rc_cn


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
    roc_analysis_train = pd.DataFrame(columns=["Dataset", "JC", "CN", "AA"])
    roc_analysis_test = pd.DataFrame(columns=["Dataset", "JC", "CN", "AA"])
    
    for i in folders:
        print(str(i))
        G_test, G_train, train_edges, test_edges, dataset_name = create_non_embedding_data(str(i))
        
        # on train data
        train_features, train_features_dataset_name = add_non_embedding_features(G_train, train_edges, dataset_name, train=True)
        plot_feature_correlation(train_features, train_features_dataset_name, train=True)
        rc_jc,rc_aa,rc_cn=calculate_roc_auc(train_features, train_features_dataset_name)
        dic_train = {"Dataset":dataset_name,"JC":rc_jc,"CN":rc_cn,"AA":rc_aa}
        roc_analysis_train = roc_analysis_train.append(dic_train, ignore_index=True)

        # on test data
        test_features, test_features_dataset_name = add_non_embedding_features(G_test, test_edges, dataset_name, train=False)
        plot_feature_correlation(test_features, test_features_dataset_name, train=False)
        rc_jc,rc_aa,rc_cn=calculate_roc_auc(test_features, test_features_dataset_name)
        dic_test = {"Dataset":dataset_name,"JC":rc_jc,"CN":rc_cn,"AA":rc_aa}
        roc_analysis_test = roc_analysis_test.append(dic_test, ignore_index=True)
    
    roc_analysis_train.to_csv('docs/roc_analysis_train.csv')
    roc_analysis_test.to_csv('docs/roc_analysis_test.csv')