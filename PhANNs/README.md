# PhANNs (Phage Artificial Neural Networks)

PhANNs is a tool to classify any phage ORF as any number of given classes. It uses an ensemble of Artificial Neural 
Networks. (PhANNs was originally designed, written by, and published by Adrian Cantu). This is a clean rewrite to
make it easier to retrain on custom datasets and specify specific feature models to train.


### Contents



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

### Train

