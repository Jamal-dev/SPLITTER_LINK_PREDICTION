import numpy as np 
import pickle
from pathlib import Path
import re



def load_numpy_data(path):
    dataset_name = re.sub('\([\w]*\)','',str(path).split('/')[-1])

    with open(f'{str(path)}/index.pkl','rb') as f:
        index = pickle.load(f)
    
    stat = []
    for i in index.keys():
        if i =='index':
            continue
        stat.append('{} for {} is: {}'.format(i,dataset_name,index[i]))

    print('\n'.join(stat))

    train = np.load(f'{str(path)}/train.txt.npy')
    train_neg = np.load(f'{str(path)}/train.neg.txt.npy')
    test = np.load(f'{str(path)}/test.txt.npy')
    test_neg = np.load(f'{str(path)}/test.neg.txt.npy')

    return(index, train, train_neg, test, test_neg)


if __name__ == '__main__':
    folders = Path('datasets_pp/original/').glob('*')
    for i in folders:
        load_numpy_data(str(i))