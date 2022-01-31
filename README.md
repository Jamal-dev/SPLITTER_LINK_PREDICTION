# SPLITTER LINK PREDICTION
```
git clone https://github.com/Jamal-dev/SPLITTER_LINK_PREDICTION.git
<br />
cd SPLITTER_LINK_PREDICTION

```
<br />
Goto SPLITTER_LINK_PREDICTION folder if the SPLITER folder is empty then:

```
git clone https://github.com/Jamal-dev/Splitter.git

```
Same with asymproj_edge_dnn_tensorFlow2 folder if it's empty:

```
git clone https://github.com/Jamal-dev/asymproj_edge_dnn_tensorFlow2.git

```
## Conda Installation
Create first an environment in conda

```
conda create -n splitter_link_pred
<br />
conda activate splitter_link_pred

```
Then you can install the requirements.txt file into the environemnt:

```
pip install -i requirements.txt

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
