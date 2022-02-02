# SPLITTER LINK PREDICTION
```
git clone https://github.com/Jamal-dev/SPLITTER_LINK_PREDICTION.git

cd SPLITTER_LINK_PREDICTION

```
<br />
Goto SPLITTER_LINK_PREDICTION:

```
git clone https://github.com/Jamal-dev/Splitter.git

```
Same with asymproj_edge_dnn_tensorFlow2 folder:

```
git clone https://github.com/Jamal-dev/asymproj_edge_dnn_tensorFlow2.git

```
## Conda Installation
Create first an environment in conda

```
conda create --name splitter_link_pred --file requirements.txt
```
Or you can do it in 3 steps
```
conda create -n splitter_link_pred

conda activate splitter_link_pred

conda install pip

```
Then you can install the requirements.txt file into the environemnt:

```
pip install -r requirements.txt

```
# Run Node2Vec
In the Link_Prediction folder:

```
python tasks/node2vecEdgeEmbeddings.py

```

# Run Splitter link prediction
In the Link_Prediction folder:

```
python tasks/splitter_edgeEmbedding.py --dataset_name CA-AstroPh --embed_dim 16

```
If you want to run all experiments as in the paper

```
python tasks/splitter_edgeEmbedding.py --all_exp 1

```

We have tested the code for Unix system, but we haven't checked for the Windows system. However, in the tasks/node2vecEdgeEmbeddings.py and tasks/splitter_edgeEmbedding.py are tested for windows system as well.
# Datasets
You can download the datasets from https://snap.stanford.edu/data/. For splitting them into the train and test sets, we used the file create_dataset_arrays.py from asymproj_edge_dnn_tensorFlow2. All files of datasets could not be uploaded because of the 50MB limit of Github. Therefore, we suggest to split the datset by using create_dataset_arrays.py file first and then place them in the dataset_pp/original/{dataset_name}, and then run the code  