# PhANNs (Phage Artificial Neural Networks)

PhANNs is a tool to classify any phage ORF as any number of given classes. It uses an ensemble of Artificial Neural 
Networks. (PhANNs was originally designed, written by, and published by Adrian Cantu). This is a clean rewrite to
make it easier to retrain on custom datasets and specify specific feature models to train.


### Contents
- [Installation](#installation)
- [How to Use](#use)
  - [Load the Data](#load)
  - [Train Model(s)](#train)
- [How to Cite This Project](#citation)


## Installation
```
git clone https://github.com/seanfahey1/PhANNs.git && \
cd PhANNs/PhANNs && \
conda env create --name phanns --file=environment.yml
```

## Use
### Load
1. `conda activate phanns`
2. Combine all fasta formatted sequences from each desired class into a single `.fasta` file. Header information for 
each sequence may be useful to the user, but is not required to be unique or present for the load step. 

3. Split each of these files into 11 roughly evenly sized `.fasta` files. I recommend the cluster-split-expand method 
from Adrian's original PhANNs paper.
   1. These files need to be prefixed with the number (eg. `tail-spike.fasta` would become `1_tail-spike.fasta`, 
   `2_tail-spike.fasta`, ..., `11_tail-spike.fasta`)

4. Define a `project_root_dir`, `train_data_dir`, and `output_data_dir` in the `config.toml` file.

5. Move the `.fasta` files to `./project_root_dir/train_data_dir/`

6. `python load.py`

7. Watch the `load.log` file for updates as the script runs. 

_Note: the load.py script will always generate arrays for all available models. Users can select a subset of arrays to 
train in the next step._ 

### Train
1. `conda activate phanns`



# Citation
Please cite as
```
Cantu, V.A., Salamon, P., Seguritan, V., Redfield, J., Salamon, D., Edwards, R.A., and Segall, A.M. (2020). PhANNs, a fast and accurate tool and web server to classify phage structural proteins. BioRxiv 2020.04.03.023523.
```
or
```
@article{Cantu_Salamon_Seguritan_Redfield_Salamon_Edwards_Segall_2020, title={PhANNs, a fast and accurate tool and web server to classify phage structural proteins}, DOI={10.1101/2020.04.03.023523}, abstractNote={<p>For any given bacteriophage genome or phage sequences in metagenomic data sets, we are unable to assign a function to 50-90% of genes. Structural protein-encoding genes constitute a large fraction of the average phage genome and are among the most divergent and difficult-to-identify genes using homology-based methods. To understand the functions encoded by phages, their contributions to their environments, and to help gauge their utility as potential phage therapy agents, we have developed a new approach to classify phage ORFs into ten major classes of structural proteins or into an “other” category. The resulting tool is named PhANNs (Phage Artificial Neural Networks). We built a database of 538,213 manually curated phage protein sequences that we split into eleven subsets (10 for cross-validation, one for testing) using a novel clustering method that ensures there are no homologous proteins between sets yet maintains the maximum sequence diversity for training. An Artificial Neural Network ensemble trained on features extracted from those sets reached a test F 1 -score of 0.875 and test accuracy of 86.2%. PhANNs can rapidly classify proteins into one of the ten classes, and non-phage proteins are classified as “other”, providing a new approach for functional annotation of phage proteins. PhANNs is open source and can be run from our web server or installed locally.</p>}, journal={bioRxiv}, publisher={Cold Spring Harbor Laboratory}, author={Cantu, Vito Adrian and Salamon, Peter and Seguritan, Victor and Redfield, Jackson and Salamon, David and Edwards, Robert A. and Segall, Anca M.}, year={2020}, month={Apr}, pages={2020.04.03.023523} }
```
