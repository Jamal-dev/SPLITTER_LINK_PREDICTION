import pandas as pd 
import networkx as nx
import json 



path = '/home/shady/Projects/GML/SPLITTER/Splitter/output/CA-HepTh_personas.json'

with open(path,'r') as f:
    data = json.load(f)

df_map = pd.DataFrame.from_dict(data, orient='index')
df_map =  df_map.rename(columns={0:'original'})
df_map['persona'] = df_map.index

G_p = nx.read_adjlist("/home/shady/Projects/GML/SPLITTER/Splitter/output/CA-HepTh_persona_adj.adjlist")

path = '/home/shady/Projects/GML/SPLITTER/Splitter/input/CA-HepTh_edges.csv'

df_G = pd.read_csv(path,index_col=0)

assert G_p.number_of_edges() == df_G.shape[0]