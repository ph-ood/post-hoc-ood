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
│  ├── plots
│  └── raw
└── README.md
```
  
## Data
- MNIST: Download and extract from https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz 
- CIFAR-10: Download and extract from https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders
- `cd code` and run `python3 data2csv.py <dataset_name>`,`<dataset_name>` can be `mnist/cifar10`
- This will create a file `data.csv` in the `data/<dataset_name>` directory
- Run `python3 mean_and_std.py <dataset_name>` to compute the channel-wise mean and std of the data and add these values to `config.py`

## Training a Classifier Model
- Add model definition to `models`
- Change the model used in `run_classifier.py`
- Run `python3 run_classifier.py <dataset_name>`
- This saves the weights of the trained model in `weights/<dataset_name>` directory

## Pretrained Scoring
- Change pretrained model path in `test_ood.py`
- Run `python3 test_ood.py -i <id_dataset_name> -o <ood_dataset_name> -s <score_name>`
- This saves scores for ID and OOD data as `.npy` files in `results/raw`

## Plots and Metrics for OOD Detection
- Run `python3 analyze_scores.py -i <id_dataset_name> -o <ood_dataset_name> -s <score_name>`
- This saves the density plots for the scores in `results/plots` 

## Running on Colab
- Create a Google Drive folder having the same root structure and upload the data
- Add a colab notebook to this folder, mount drive
- `cd` to created folder and run the `.py` scripts from notebook command line
- Sync to and from Drive when needed
