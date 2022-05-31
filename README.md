# Regularized Simple Graph Convolution for link prediction

by [Patrick Pho (Phuong Pho)](https://scholar.google.com/citations?user=yuvA4AkAAAAJ&hl=en) and [Alexander V. Mantzaris](https://scholar.google.com/citations?user=8zP4vSQAAAAJ&hl=en)

This repo is an official implementation of the Regularized Simple Graph Convolution (SGC) for link prediction task in our paper - "Link prediction with Simple Graph Convolution and regularized Simple Graph Convolution".

We adopt the flexible regularization scheme introduced in our previous [work](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00366-x) - "Regularized Simple Graph Convolution (SGC) for improved interpretability of large datasets" - for link predictor module's weight vector. The $L_1$ term reduces the number of components of the weight vectors, the $L_2$ term controls the overall size of the weight vectors. The proposed framework produces sparser set of fitted weights highlighting important edge embeddings that define link likelihood.

## Prerequisites
The dependencies can be install via:
```
pip install -r requirement.txt
```  

For GPU machine, please refer to official instruction to install suitable version of `pytorch` and `dgl`:
- [PyTorch](https://pytorch.org/)
- [Deep Graph Library - DGL](https://www.dgl.ai/pages/start.html)

## Data
Three citation datasets (Cora, Citeseer, and Pubmed) are available for user to experiment with our framework. These datasets are included in DGL package and can be selected by specifying `--dataset` argument (see example in the **Usage** section).

We also provide utility function `import_data` to assist users in importing their own dataset.

## Usage
### Train model
An example of incorporating $L_1 = 0.5, L_2 = 1.0,$ into SGC fitted on Cora dataset is:
```
python main.py --dataset cora --L1 0.5 --L2 1
```

Use `--save-trained` to save trained model for inference. The trained model is save in `./checkpoints`
```
python main.py --dataset cora --L1 0.5 --L2 1 --save-trained
```

Other useful options for training:
- `--early-stop`: turn on early stopping to reduce overfitting. Default metric is loss
- `--hist-print`: print training history at every *t* epoch