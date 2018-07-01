# MemGCN

# Memory-Augmented GCN (MemGCN)

## Overview

This repository contains TensorFlow code for implementing Memory-Augmented Graph Convolutional Network multi-modal data learning in healthcare. The rationale behind the model is that both patient health records and neuroimages are important for disease understanding because of their complementary aspects of diseases.  

<p align="center"><img src="images/overview.png" alt="Memory-base Graph Convolutional Network for brain connectivity graphs with clinical records" width="500"></p>
<p align="center"><img src="images/GCN.png" alt="Multi-View GCN" width="600"></p>

In detail, the proposed method MemGCN is a matching network embeds multi-hop memory-augmented graph convolutions and can be trained in an end-to-end fashion with stochastic optimization. The brain connectivity graphs (DTI) are transformed by graph convolutional networks into representation, while the external memory mechanism is in charge of iteratively (multiple hops) reading clinical sequences and choosing what to retrieve from memories so that the representations learned by graph convolution can be augmented.

MemGCN provides a learning strategy for multi-modality data with sequential and graph structure in general scenarios. The code is documented and should be easy to modify for your own applications.      


## Requirements
This package has the following requirements:
* An NVIDIA GPU.
* `Python3.x`
* [TensorFlow](https://github.com/tensorflow/tensorflow)

## Usage
To run MemGCN on your data, you need to: change the function of loading data in utils.py; set hyperparameters for MemGCN in memgcn.sh; run the shell script memgcn.sh
```bash
bash memgcn.sh
```
