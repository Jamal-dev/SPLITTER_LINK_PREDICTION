import json
from pathlib import Path
import pandas as pd
import argparse
import os
from subprocess import call

file_runing_dir = os.path.dirname(os.path.abspath(__file__))
def check_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_json", type=str, help="Please enter the path for the json file of the paper",
                        nargs='?', default=Path(f"{file_runing_dir}/../paper_results.json"))

    args = parser.parse_args()
    jsn_path  = args.paper_json
    # paths for the results files

    
    return jsn_path


def results_paths():
    paths = {
              "Node2Vec":[Path(f'{file_runing_dir}/../Results/node2Vec_train_edge_pp_embed.csv'),
                       Path(f'{file_runing_dir}/../Results/node2Vec_test_edge_pp_embed.csv')],
             "Asymmetric":[Path(f'{file_runing_dir}/../Results/Asymmetric_trainScore_edge_embed.csv'),
                       Path(f'{file_runing_dir}/../Results/Asymmetric_testScore_edge_embed.csv')],
             "Splitter":[Path(f'{file_runing_dir}/../Results/splitter_trainScore_edge_embed.csv'),
                       Path(f'{file_runing_dir}/../Results/splitter_testScore_edge_embed.csv')],
             "Non-Embedding":[Path(f'{file_runing_dir}/../Results/Non-Embedding_org_test.csv'),
                       Path(f'{file_runing_dir}/../Results/Non-Embedding_org_train.csv')],
             "Non-Embedding-persona":[Path(f'{file_runing_dir}/../Results/Non-Embedding_pers_test.csv'),
                       Path(f'{file_runing_dir}/../Results/Non-Embedding_pers_train.csv')]
                       }
    return paths

def read_nonEmbed(path,paper_data,method):
    if method == "Non-Embedding":
        comp_c = 'org'
    elif method == "Non-Embedding-persona":
        comp_c = 'pers'
    df = pd.read_csv(path)
    df = df.drop("Unnamed: 0",1)
    df.set_index('Dataset', inplace=True)
    columns = [c for c in df.columns]
    indices = [i for i in df.index]
    df_result = pd.DataFrame(columns=columns,index=indices)
    for c in columns:
        for i in indices:
            df_result[c][i] = (df[c][i] - paper_data[c][str(i)][comp_c])/paper_data[c][str(i)][comp_c] 
    temp = path.name[0:-4]
    sv_path = path.parent / "diff_with_paper" / f"{temp}_diff.csv"
    df_result.to_csv(sv_path)
    lt_path = Path(f'{file_runing_dir}/../utils/latex_longtable.py')
    cmd_run = ['python',str(lt_path),'--input',str(sv_path)]
    call(cmd_run)


def read_file(path,paper_data,method):
    if 'Non-Embedding' in method:
        read_nonEmbed(path,paper_data,method)
        return 
    df = pd.read_csv(path)
    if 'Embedding Dimension' in df.columns:
        df.set_index('Embedding Dimension', inplace=True)
    else:
        df=df.rename(columns = {'Unnamed: 0':'Embedding Dimension'})
        df.set_index('Embedding Dimension', inplace=True)
    columns = [c for c in df.columns]
    indices = [i for i in df.index]
    df_result = pd.DataFrame(columns=columns,index=indices)
    for c in columns:
        for i in indices:
            df_result[c][i] = (df[c][i] - paper_data[method][c][str(i)])/paper_data[method][c][str(i)] 
    temp = path.name[0:-4]
    sv_path = path.parent / "diff_with_paper" / f"{temp}_diff.csv"
    df_result.to_csv(sv_path)
    lt_path = Path(f'{file_runing_dir}/../utils/latex_longtable.py')
    cmd_run = ['python',str(lt_path),'--input',str(sv_path)]
    call(cmd_run)

def main():
    jsn_path = check_args()
    with open(jsn_path) as f:
        paper_data = json.loads(f.read())
        f.close()
    all_paths = results_paths()
    for method,paths in all_paths.items():
        for path in paths:
            read_file(path,paper_data,method)
            if 'test' in path.name:
                testOrTrain = 'test'
            else:
                testOrTrain = 'train'
            print(f'{method} for {testOrTrain} is done!')

if __name__ == "__main__":
    main()