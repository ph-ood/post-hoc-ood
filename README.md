# CS-726 Course Project
  
Root directory structure:
```
.
├── code
├── data
│  ├── cifar10
│  │  ├── test
│  │  └── train
│  └── mnist
│     ├── test
│     └── train
├── weights
│  ├── cifar10
│  └── mnist
├── results
└── README.md

```
  
## Data
- MNIST: Download and extract from https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz 
- CIFAR-10: Download and extract from https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders
- `cd code` and run `python3 data2csv.py <data_name>`. `<data_name>` can be `mnist/cifar10`. This will create a `data.csv` file in the `data/<data_name>` directory.
