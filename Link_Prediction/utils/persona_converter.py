import pandas as pd 
import networkx as nx
import json 
from pathlib import Path
import re
import numpy as np


def load_persona(path_json, path_persona, path_original_edges, dataset_name):

    # read the persona graph json
    print(f'reading {dataset_name} persona graph json ...')
    with open(path_json, 'r') as f:
        persona_graph_map = json.load(f)

    # convert the persona graph json to pandas dataframe
    print(f'converting {dataset_name} persona graph json to pandas dataframe ...')
    df_map = pd.DataFrame.from_dict(persona_graph_map, orient='index')
    df_map =  df_map.rename(columns={0:'original'})
    df_map['persona'] = df_map.index
    print(df_map.head())

    # load persona graph adjacency matrix
    print(f'loading {dataset_name} persona graph adjacency matrix ...')
    G_p = nx.read_adjlist(path_persona)
    print('number of nodes in persona graph: {} | numer of edges in persona graph: {}'.format(G_p.number_of_nodes(), G_p.number_of_edges()))
    nx.write_gexf(G_p, f"graphs_pp/persona/{dataset_name}.gexf")
    print(f'{dataset_name} graph saved as gexf')
    
    # create text format of edge list
    edge_list = nx.to_pandas_edgelist(G_p)
    edge_list = edge_list.sort_values(['source', 'target'], inplace=False)
    edge_list = edge_list.reset_index(drop=True)
    numpy_array = edge_list.to_numpy(dtype=np.int32)
    print(numpy_array)
    np.savetxt(f"datasets_raw/persona/{dataset_name}.txt", numpy_array, fmt='%d')

    # load original edges
    print(f'loading {dataset_name} original edges ...')
    df_G = pd.read_csv(path_original_edges,index_col=0)

    # test if the number of edges in original and persona graph are equal
    print(f'checking if the number of edges in original and persona graph are equal ...')
    assert G_p.number_of_edges() == df_G.shape[0]
    print('number of edges in original and persona graph are equal!')


if __name__ == "__main__":
    folders = Path('/home/shady/Projects/GML/SPLITTER/Splitter/output/').glob('*')
    for folder in folders:
        dataset_name = re.sub('\([\w]*\)','',str(folder).split('/')[-1])
        path_json = str(folder) + '/' + dataset_name + '_personas.json'
        path_persona = str(folder) + '/' + dataset_name + '_persona_adj.adjlist'
        path_original_edges = f'/home/shady/Projects/GML/SPLITTER/Splitter/input/{dataset_name}/' + dataset_name + '_edges.csv'
        load_persona(path_json, path_persona, path_original_edges, dataset_name)

