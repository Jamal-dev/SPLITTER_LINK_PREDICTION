import sys
sys.path.insert(0, ".")

from pathlib import Path
import pandas as pd 
import networkx as nx
import json 
from pathlib import Path
import re
import numpy as np
from math import log
from tqdm import tqdm
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import os.path

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
        aa = sum(1/log(len(set(g[w]))) for w in cn_dir(g, i, j))
    except:
        aa = 0
    return(aa)

def load_persona(dataset_name, path_adj, path_train, path_json):

    dir_graph = ['soc-epinions','wiki-vote']
    undir_graph = ['ca-astroph','ca-hepth','ppi']

    # dataset_name = re.sub('\([\w]*\)','',str(path).split('/')[-1])

    if dataset_name.lower() in dir_graph:
        G_p = nx.read_adjlist(path_adj, create_using=nx.DiGraph())
    elif dataset_name.lower() in undir_graph:
        G_p = nx.read_adjlist(path_adj, create_using=nx.Graph())
    else:
        raise ValueError('Invalid Graph')

    # convert persona graph to dataframe
    df_P = nx.to_pandas_edgelist(G_p, source='Source', target='Target')
    df_P = df_P[df_P['Source'] != df_P['Target']]
    df_P[['Source','Target']] = df_P[['Source','Target']].astype(int)
    df_P = df_P.sort_values(['Source','Target'])
    df_P['Edges'] = list(zip(df_P.Source, df_P.Target))
    df_P = df_P.drop(columns=['Source','Target'], inplace=False)
    df_P['Connection'] = 1
    df_P = df_P.reset_index(drop=True)

    # load train edges
    df_G = pd.read_csv(path_train)
    df_G[['Source','Target']] = df_G['Edges'].str.extractall('(\d+)').unstack().loc[:,0]
    df_G = df_G[df_G['Source'] != df_G['Target']]
    df_G[['Source','Target']] = df_G[['Source','Target']].astype(np.int32)
    df_G = df_G.sort_values(['Source','Target'])
    df_G['new_edges'] = list(zip(df_G.Source, df_G.Target))
    df_G = df_G.drop(columns=['Source','Target','Edges'], inplace=False)
    df_G = df_G[['new_edges','Connection']]

    # load persona json
    with open(path_json, 'r') as f:
        persona_graph_map = json.load(f)

    # normalizing the json sile
    persona_graph_map_list = list(persona_graph_map.items())

    personas = {}
    for x, y in persona_graph_map_list:
        if y in personas:
            personas[y].append(x)
        else:
            personas[y] = [x]

    
    return(dataset_name, G_p, df_P, df_G, personas)


def add_non_embedding_features(dataset_name,G_p,df_P):
    
    if G_p.__class__.__name__ == 'DiGraph':
        print(f'{dataset_name} graph is directed')
        cn, jc, aa = [], [], []
        for e in tqdm(df_P['Edges'].tolist()):
            cn.append(len(cn_dir(G_p, str(e[0]), str(e[1]))))
            jc.append(jc_dir(G_p, str(e[0]), str(e[1])))
            aa.append(aa_dir(G_p, str(e[0]), str(e[1])))

        df_P['CN'] = cn
        df_P['JC'] = jc
        df_P['AA'] = aa


    elif G_p.__class__.__name__ == 'Graph':
        print(f'{dataset_name} graph is undirected')
        cn, jc, aa = [], [], []
        for e in tqdm(df_P['Edges'].tolist()):
            cn.append(len(cn_undir(G_p, str(e[0]), str(e[1]))))
            jc.append(jc_undir(G_p, str(e[0]), str(e[1])))
            aa.append(aa_undir(G_p, str(e[0]), str(e[1])))

        df_P['CN'] = cn
        df_P['JC'] = jc
        df_P['AA'] = aa
    

    else:
        raise ValueError('Invalid Graph')

    return(dataset_name, df_P)


def map_to_original(dataset_name, df_G, personas, G_p):
    for index, row in tqdm(df_G.iterrows()):
        u_val = personas[int(row['new_edges'][0])]
        v_val = personas[int(row['new_edges'][1])]
        perms = list(itertools.product(list(u_val), list(v_val)))
        
        jc, aa, cn = [], [], []
        for i in perms:
            jc.append(jc_dir(G_p, i[0], i[1]))
            aa.append(aa_dir(G_p, i[0], i[1]))
            cn.append(len(cn_dir(G_p, i[0], i[1])))


        df_G.loc[index, 'JC'] = max(jc)
        df_G.loc[index, 'AA'] = max(aa)
        df_G.loc[index, 'CN'] = max(cn)    

    scaler = MinMaxScaler()
    dataset_np = scaler.fit_transform(df_G[['Connection', 'JC', 'AA', 'CN']])
    dataset = pd.DataFrame(dataset_np, index=df_G.index ,columns=['Connection', 'JC', 'AA', 'CN'])

    return(dataset_name, dataset)


def calculate_roc_auc(dataset_name, dataset):
    
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

    print(f"ROC-score:\n Jacard Coefficent:{rc_jc:1.3f}, Admic Adar:{rc_aa:1.3f}, Common neighbours:{rc_cn:1.3f}")
    
    return rc_jc,rc_aa,rc_cn


if __name__ == "__main__":
    folders = Path('datasets_pp/persona/').glob('*')
    roc_analysis_train = pd.DataFrame(columns=["Dataset", "JC", "CN", "AA"])

    for folder in folders:
        print(str(folder))
        dataset_name = re.sub('\([\w]*\)','',str(folder).split('/')[-1])
        path_adj = str(folder) + '/' + dataset_name + '_adj.adjlist'
        path_train = f'datasets_pp/original/{dataset_name}/' + dataset_name + '_train_edges.csv'
        path_json = str(folder) + '/' + dataset_name + '_personas.json'

        if os.path.exists(path_adj) and os.path.exists(path_train) and os.path.exists(path_json):
            dataset_name, G_p, df_P, df_G, personas = load_persona(dataset_name, path_adj, path_train, path_json)
            dataset_name, df_P = add_non_embedding_features(dataset_name, G_p, df_P)
            dataset_name, dataset = map_to_original(dataset_name, df_G, personas, G_p)
            rc_jc, rc_aa, rc_cn = calculate_roc_auc(dataset_name, dataset)
            dic_train = {"Dataset":dataset_name,"JC":rc_jc,"CN":rc_cn,"AA":rc_aa}
            roc_analysis_train = roc_analysis_train.append(dic_train, ignore_index=True)

    roc_analysis_train.to_csv('docs/roc_analysis_train_persona.csv')
        

# def load_persona(path_json, path_persona, path_original_edges, dataset_name):

#     # read the persona graph json
#     print(f'reading {dataset_name} persona graph json ...')
#     with open(path_json, 'r') as f:
#         persona_graph_map = json.load(f)

#     # convert the persona graph map to pandas dataframe
#     print(f'converting {dataset_name} persona graph json to pandas dataframe ...')
#     df_map = pd.DataFrame.from_dict(persona_graph_map, orient='index')
#     df_map =  df_map.rename(columns={0:'original'})
#     df_map['persona'] = df_map.index
#     print(df_map.head())

#     # load persona graph adjacency matrix
#     print(f'loading {dataset_name} persona graph adjacency matrix ...')
#     G_p = nx.read_adjlist(path_persona)
#     print('number of nodes in persona graph: {} | numer of edges in persona graph: {}'.format(G_p.number_of_nodes(), G_p.number_of_edges()))
#     nx.write_gexf(G_p, f"graphs_pp/persona/{dataset_name}.gexf")
#     print(f'{dataset_name} graph saved as gexf')
    
#     # create text format of edge list
#     edge_list = nx.to_pandas_edgelist(G_p)
#     edge_list = edge_list.sort_values(['source', 'target'], inplace=False)
#     edge_list = edge_list.reset_index(drop=True)
#     numpy_array = edge_list.to_numpy(dtype=np.int32)
#     print(numpy_array)
#     np.savetxt(f"datasets_raw/persona/{dataset_name}.txt", numpy_array, fmt='%d')

#     # load original edges
#     print(f'loading {dataset_name} original edges ...')
#     df_G = pd.read_csv(path_original_edges,sep='\t',skiprows=4,header=None,names=['source','target'])
#     print(df_G.head())

#     # test if the number of edges in original and persona graph are equal
#     print(f'checking if the number of edges in original and persona graph are equal ...')
#     assert G_p.number_of_edges() == df_G.shape[0]
#     print('number of edges in original and persona graph are equal!')