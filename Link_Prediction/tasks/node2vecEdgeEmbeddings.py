import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from utils.load_numpy_dataset import load_numpy_data
import re
from node2vec import Node2Vec
# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder
import os
from pathlib import Path
from tqdm import tqdm

file_runing_dir = os.path.dirname(os.path.abspath(__file__))
class node2vec_edgeEmb:
    def __init__(self,p=1.0,q=1.0,dimensions=64,num_walks=10,walk_length=40,window_size=5,random_state=42,workers=1,path = Path(f'"{file_runing_dir}/../datasets_pp/original/ca-hepth/"')):
        self.p = p 
        self.q = q 
        self.dimensions = dimensions 
        self.num_walks = num_walks 
        self.walk_length = walk_length
        self.window_size = window_size
        self.workers = workers
        self.random_state = random_state
        self.path = path

    def graph_edge(self,data_train,data_train_ng,if_neg=True):
        df = pd.DataFrame(data_train,columns=['source','target'])
        df['weight'] = np.ones(shape=(data_train.shape[0],))
        df2=pd.DataFrame(data_train_ng,columns=['source','target'])
        if if_neg:
            df2['weight'] = np.zeros(shape=(data_train_ng.shape[0],))
        else:
            df2['weight'] = np.ones(shape=(data_train_ng.shape[0],))
        df = df.append(df2)
        # shuffle the DataFrame rows
        df = df.sample(frac = 1,random_state=self.random_state)
        # reset the index
        df = df.reset_index(drop=True)
        G = nx.from_pandas_edgelist(df, create_using=nx.Graph(),edge_attr=True)
        return G,df

    def node2vec_EdgeEmbeder(self,graph):
        nv = Node2Vec(graph, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers,weight_key='weight') 
        model = nv.fit(window=self.window_size, min_count=1, batch_words=4)
        edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
        return edges_embs

    def node2Vec_dataset_creation(self,G,edge_embed):
        X=[]
        y = []
        for u,v in G.edges():
            lbl = G[u][v]["weight"]
            y.append(lbl)
            
            X.append(edge_embed[str(u),str(v) ])
        X = np.asarray(X)
        return X,y

    def fit(self):

        dataset_name = re.sub('\([\w]*\)','',str(self.path).split('/')[-1])
        if dataset_name=='':
            dataset_name = re.sub('\([\w]*\)','',str(self.path).split('/')[-2])

        self.dataset_name = dataset_name
        index, train, train_neg, test, test_neg = load_numpy_data(self.path)
        
        # creating train dataset
        print(self.dataset_name)
        # train and test set
        G_train,_=self.graph_edge(train,train_neg)
        G_test,_=self.graph_edge(test,test_neg)
        G_embed,_=self.graph_edge(train,test, if_neg=False)
        # edge embedding
        edge_embed=self.node2vec_EdgeEmbeder(G_embed)
        # dataset creation
        X_train, y_train = self.node2Vec_dataset_creation(G_train,edge_embed)
        X_test , y_test  = self.node2Vec_dataset_creation(G_test,edge_embed)
        # classifier
        clf = LogisticRegression(random_state=self.random_state).fit(X_train, y_train)
        # train score
        self.score_train = roc_auc_score(y_true=y_train, y_score=clf.predict(X_train),  average='micro')
        # test score
        self.score_test = roc_auc_score(y_true=y_test, y_score=clf.predict(X_test),  average='micro')

        print(f"Dataset: {self.dataset_name}\n ROC-Score (train): {self.score_train}\n ROC-Score (test): {self.score_test}")        
        
        return self.score_train,self.score_test, self.dataset_name

def main():
    folder_dir = Path(f'{file_runing_dir}/../datasets_pp/original')
    # col_names = ["CA-HepTh","ca-AstroPh","ppi","soc-epinions","wiki-vote"]
    col_names = ["CA-AstroPh","Wiki-Vote","soc-Epinions"]
    dataset_dir = [folder_dir / c for c in col_names]
    
    
    dimensions = [8,16,32,64,128]
    df_train = pd.DataFrame(columns=col_names, index=dimensions)
    df_test = pd.DataFrame(columns=col_names, index=dimensions)
    for d_dir in tqdm(dataset_dir):
        for d in tqdm(dimensions):
            cl = node2vec_edgeEmb(path= d_dir, dimensions=d)
            train_score,test_score,name = cl.fit()
            df_test[name][d] = test_score
            df_train[name][d] = train_score
    print("Train:\n",df_train)
    print("Test:\n",df_test)
    df_train.to_csv(f'"{file_runing_dir}/../Results/node2vec_edge_pp_embed_train.csv"')
    df_test.to_csv(f'"{file_runing_dir}/../Results/node2vec_edge_pp_embed_test.csv"')


if __name__=="__main__":
    main()
