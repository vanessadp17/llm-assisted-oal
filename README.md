## Setup and Dependencies

1. Create and activate a conda environment as follows: 
```
conda create -n laser-env python
conda activate laser-env
```
2. Install dependencies: 
```
pip install -r requirements.txt
```
3. Modify the dataloader.py file in the torch.util.data.Dataloader source code within the conda environment as described [here](https://github.com/ningkp/LfOSA/issues/4).

### CIFAR-10
For the CIFAR-10 experiment with a mismatch ratio of 20%, run:

```
python main.py --query-strategy laser --known-class 2 --dataset cifar10 --K-td 15  --gpu 0
```

### CIFAR-100

For the CIFAR-100 experiment with a mismatch ratio of 20%, run:

```
python main.py --query-strategy laser --known-class 20 --dataset cifar100 --K-td 4 --gpu 0
```

## Acknowledgements

This code is built upon [EOAL](https://github.com/bardisafa/EOAL) repository.