# SPLITTER LINK PREDICTION
git clone https://github.com/Jamal-dev/SPLITTER_LINK_PREDICTION.git
<br />
cd SPLITTER_LINK_PREDICTION
<br />
Goto SPLITTER_LINK_PREDICTION folder if the SPLITER folder is empty then:
<br />
git clone https://github.com/Jamal-dev/Splitter.git
<br />
Same with asymproj_edge_dnn_tensorFlow2 folder if it's empty:
<br />
git clone https://github.com/Jamal-dev/asymproj_edge_dnn_tensorFlow2.git
<br />
## Conda Installation
Create first an environment in conda
<br />
conda create -n splitter_link_pred
<br />
conda activate splitter_link_pred
<br />
Then you can install the requirements.txt file into the environemnt:
<br />
pip install -i requirements.txt
<br />
# Run Node2Vec
In the Link_Prediction folder:
<br />
python tasks/node2vecEdgeEmbeddings.py
<br />

# Run Splitter link prediction
In the Link_Prediction folder:
<br />
python tasks/splitter_edgeEmbedding.py --dataset_name CA-AstroPh --embed_dim 16
<br />
If you want to run all experiments as in the paper
<br />
python tasks/splitter_edgeEmbedding.py --all_exp 1
