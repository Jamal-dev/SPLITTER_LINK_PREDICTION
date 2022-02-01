import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def check_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Write path for csv file",
                        nargs='?', default="assymetric_testScore_edge_embed.csv")
    parser.add_argument("--out", type=str, help="Plese enter the path for the output txt file. If keeps empty it will be same as input",
                        nargs='?', default="-", const="-")

    args = parser.parse_args()
    path = Path(args.input)
    out_path = Path(args.out)
    tab_printer(args)
    return path, out_path
    
    
    
    return path, out_path

def main():
    path, out_path = check_args()
    df =pd.read_csv(path)
    df=df.rename(columns = {'Unnamed: 0':'Embedding \\\\ Dimension'})
    len_c = len(df.columns)
    name = path.name.split('_')[0].capitalize()
    if 'test' in path.name:
        testOrTrain = 'test'
    else:
        testOrTrain = 'train'
    
    header = r'%' + f'Table for {name}: {testOrTrain}' + '\n'
    header = header  + r'\begin{longtable}[]{@{}'
    for i in range(len_c):
        header = header + '\n'+r'>{\raggedright\arraybackslash}p{2cm}'
    header = header + '@{}}'
    header = header + f'\n\caption{{{name} results for the link prediction on the {testOrTrain} edges \label{{tab: {name}-{testOrTrain}-results}}}}'
    header = header + r"\tabularnewline"
    header = header + '\n' + r'\toprule' + '\n\endhead'
    for i,c in enumerate(df.columns):
        if i==0:
            header = header + '\makecell{' + c.capitalize() + '} & '
        else:
            if i==len_c-1:
                header = header + f'{c.capitalize()} \\\\'
            else:
                header = header + f'{c.capitalize()} & '
    body = '\n'
    for index,row in df.iterrows():
        
        for r in row:
            if r == row[-1]:
                body = body + '\\numS{' + str(r) + '}' + ' \\\\ \n'
            else:
                if r == row[0]:
                    body = body  + str(int(r))  + ' & '
                else:
                    body = body + '\\numS{' + str(r) + '}' + ' & '
    fotter = '\n'+r'\bottomrule'
    fotter = fotter + '\end{longtable}'

    if str(out_path)=="-":
        temp = str(path)
        out_path = Path(temp[0:-4] + '.txt')
        out_path = Path(out_path.parent / 'latex' / out_path.name)
        print(f"The Modified output path:\n {out_path}")
    with open(out_path,'w') as f:
        f.write(header + body + fotter)
        f.close()

if __name__=='__main__':
    main()