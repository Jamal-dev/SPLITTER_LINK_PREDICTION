import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from utils.load_numpy_dataset import load_numpy_data
import itertools
import os
from subprocess import call
import json
from csv import DictWriter
import argparse

def if_not_exists(filename,train,test,embed_dim):
    """
    Create the embedding files for the Splitter if
    does not exist already.
    Returns
    -------
    True if the file was not on the path,
    False if it already existed
    """
    if not os.path.exists(filename):
        print(f"{filename} did not exit. Downloading has started ...")
        create_embed(train,test,embed_dim)
        print(f"{filename} has downloaded...")
        return True
    print(f"{filename} is already there...")
    return False

def mappedNodes_corresponding_originalNodes(map_og):
    org_nodes = set(map_og.values())

    # mapped nodes are the personas nodes and personas nodes are unique
    mappedNodes_fromOrg = {}
    for n in org_nodes:
        mappedNodes_fromOrg[n] = [k for k in map_og.keys() if map_og[k] == n]
    return mappedNodes_fromOrg


def load_embed(dataset_name):
    with open(f"datasets_pp/persona/{dataset_name}_personas.json") as f:
        data = json.loads(f.read())
    
    map_og = {}
    # k: personas node (they are unique)
    # v: original nodes (they are not unique)
    for k,v in data.items():
        # mapped nodes to the original
        map_og[int(k)] = int(v)
    return map_og

def create_embed(train,test,embed_dim):
    data = np.concatenate([train,test],axis=0)
    np.savetxt(f"../Splitter/input/{dataset_name}_{embed_dim}.csv", data.astype(int),  fmt='%i',delimiter=",")
    print(f'Training file for {dataset_name} has been and saved in Spliter/input')
    csv_name = f"{dataset_name}_{embed_dim}.csv" 
    emb_name = f"{dataset_name}_{embed_dim}_embedding.csv"
    pers_name = f"{dataset_name}_{embed_dim}_personas.json"
    cmd_input = ["python", "../Splitter/src/main.py", 
        "--dimensions", str(embed_dim), 
            "--edge-path", f"../Splitter/input/{csv_name}",  
         "--embedding-output-path", f"datasets_pp/persona/{emb_name}",  
             "--persona-output-path", f"datasets_pp/persona/{pers_name}"]
    call(cmd_input,timeout=None)
    print("Splitter embedding has been writen successfully!")

def load_data(file_path,dataset_name,embed_dim):
    path = f"datasets_pp/original/{dataset_name}"
    index, train, train_neg, test, test_neg = load_numpy_data(path)
    if_not_exists(file_path,train,test,embed_dim)
    return train, train_neg, test, test_neg
    

def graph_edge(data_train,data_train_ng,if_neg=True):
    df = pd.DataFrame(data_train,columns=['source','target'])
    df['weight'] = np.ones(shape=(data_train.shape[0],))
    df2=pd.DataFrame(data_train_ng,columns=['source','target'])
    if if_neg:
        df2['weight'] = np.zeros(shape=(data_train_ng.shape[0],))
    else:
        df2['weight'] = np.ones(shape=(data_train_ng.shape[0],))
    df = df.append(df2)
    # shuffle the DataFrame rows
    df = df.sample(frac = 1,random_state=random_state)
    # reset the index
    df = df.reset_index(drop=True)
    G = nx.from_pandas_edgelist(df, create_using=nx.Graph(),edge_attr=True)
    return G,df
def append_result(result):
    headersCSV = ['Dataset Name','Embedding Dimension','ROC Train score','ROC Test score']
    filename = 'Results/splitter_results.csv'
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=headersCSV,index = None)
        df.to_csv(filename, index = False) 
    else:     
        with open(filename, 'a', newline='') as f_object:
            # Pass the CSV  file object to the Dictwriter() function
            # Result - a DictWriter object
            dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
            # Pass the data in the dictionary as an argument into the writerow() function
            dictwriter_object.writerow(result)
            # Close the file object
            f_object.close()
def check_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Please enter the dataset name. For the corresponding name it should have directory in datasets_pp/original/dataset_name, and this directory must contain train and test files",
                        nargs='?', default="CA-AstroPh", const="CA-AstroPh")
    parser.add_argument("--embed_dim", type=int, help="Plese enter the dimension for embedding",
                        nargs='?', default=32, const=32)
    parser.add_argument("--random_state", type=int, help="Plese enter the random state",
                        nargs='?', default=42, const=42)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    embed_dim = args.embed_dim
    global random_state
    random_state = args.random_state
    
    
    return dataset_name, embed_dim


"""
    Splitter Methods
"""
def aggregate(u,v):
    try:
        res = np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))
    except:
        res = 0.0
    return res
def edge_embed(u,v,mappedNodes_fromOrg,df_emb):
    # map_og: mapped nodes/personas to the original
    # u,v are the original nodes
    # nodes1 contains all personas nodes for u
    nodes1 = mappedNodes_fromOrg[u]
    # nodes2 contains all personas nodes for v
    nodes2 = mappedNodes_fromOrg[v]
    all_prod = []
    # here u1,v1 are the nodes form personas
    for u1,v1 in itertools.product(nodes1, nodes2):
        # df_emb containts the personas nodes
        e1 = df_emb.iloc[u1,1:]
        e2 = df_emb.iloc[v1,1:]
        all_prod.append(aggregate(e1,e2))
    
    if all_prod:
        res = np.max(all_prod)
    else:
        res = 0.0
    return res
    
def splitter_edgeEmbedding(G,mappedNodes_fromOrg,df_emb):
    X=[]
    y = []
    for u,v in G.edges():
        lbl = G[u][v]["weight"]
        y.append(lbl)
        
        X.append(edge_embed(u,v,mappedNodes_fromOrg,df_emb))
    X = np.asarray(X)
    return X,y

        
def main(dataset_name,embed_dim):
    
    
    file_path = f"datasets_pp/persona/{dataset_name}_embedding.csv"
    train, train_neg, test, test_neg = load_data(file_path,dataset_name,embed_dim)
    G_train,df_train =graph_edge(train,train_neg)
    G_test,df_test =graph_edge(test,test_neg)
    
    df_emb = pd.read_csv(file_path)

    # Loading the Splitter embeddings
    map_og = load_embed(dataset_name)
    print("Splitter edge embedding have been loaded from json file")

    # Grouping personas nodes corresponding to the original nodes
    mappedNodes_fromOrg = mappedNodes_corresponding_originalNodes(map_og)

    # Embeddings for train and test graphs
    X_train,y_train = splitter_edgeEmbedding(G_train,mappedNodes_fromOrg,df_emb)
    X_test,y_test = splitter_edgeEmbedding(G_test,mappedNodes_fromOrg,df_emb)
    print("X_train, X_test have been prepared")

    # expanding dimmension of Feature vector if [1] not exists
    try:
        X_train.shape[1]
    except:
        X_train = np.expand_dims(X_train,axis=1)
        X_test = np.expand_dims(X_test,axis=1)

    # training on logistic classifier
    clf = LogisticRegression(random_state=random_state).fit(X_train, y_train)

    # train score
    score_train = roc_auc_score(y_true=y_train, y_score=clf.predict(X_train),  average='micro')
    # test score
    score_test = roc_auc_score(y_true=y_test, y_score=clf.predict(X_test),  average='micro')

    print(f"\n ROC-Score (train): {score_train}\n ROC-Score (test): {score_test}")

    # saving the results
    result = {'Dataset Name':dataset_name,'Embedding Dimension':embed_dim,'ROC Train score':score_train,'ROC Test score':score_test}
    append_result(result)
    print("Score has been appended")
    


if __name__=="__main__":
    dataset_name, embed_dim = check_args()
    main(dataset_name,embed_dim)
    