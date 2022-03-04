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
- `cd code` and run `python3 data2csv.py <dataset_name>`,`<dataset_name>` can be `mnist/cifar10`
- This will create a file `data.csv` in the `data/<dataset_name>` directory

## Training a Classifier Model
- `cd code`, add model definition to `models`
- Change the model used in `run_classifier.py`
- Run `python3 run_classifier.py <dataset_name>`
- This saves the weights of the trained model in `weights/<dataset_name>` directory

## Running on Colab
- Create a Google Drive folder having the same root structure and upload the data
- Add a colab notebook to this folder, mount drive
- `cd` to created folder and run the `.py` scripts from notebook command line
- Sync to and from Drive when needed
