# Dual-View-Expansion
Dual-View-Experiments (DDI, DTA, PPI)

Gyoung Jin Park, Dasom Noh, Gyounyoung Heo, Yeongyeong Son, Sunyoung Kwon

## Installation

```sh
conda create -n grapose python=3.12 -y
conda activate grapose

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html
pip install rdkit deepchem tqdm 

# consider pH environments, logger
pip install dimorphite-dl knockknock
conda install -c conda-forge pymol-open-source
```

---
## Datasets  <a name="datasets"></a>

source [TDC](https://tdcommons.ai/)

## Contact (Questions/Bugs/Requests)
Please submit a GitHub issue or contact my [Email](rudwls2717@pusan.ac.kr)

## Acknowledgements
Thank you for our [Laboratory](https://www.k-medai.com/).

If you find this code useful, please consider citing my work.