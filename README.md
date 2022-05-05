# CS-726 Course Project
  
Example directory structure:
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
- MNIST (`mnist`): Download and extract from https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz 
- CIFAR-10 (`cifar10`): Download and extract from https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders
- Fashion-MNIST (`fmnist`): Download and extract from https://github.com/DeepLenin/fashion-mnist_png/raw/master/data.zip
- Run `python3 data2csv.py <dataset_name>`,`<dataset_name>` can be `mnist/cifar10`
- This will create a file `data.csv` in the `data/<dataset_name>` directory
- Run `python3 mean_and_std.py <dataset_name>` to compute the channel-wise mean and std of the data and add these values to `config.py`

## Data Subsets
- Run `python3 split_data.py -d <dataset_name> -s <subset>`
- This creates a new dataset in the `data/` directory with the name `<dataset_name>_<ext>` with only the data from the specified subset
- For example, `python3 split_data -d mnist -s "3,5,6,9"` creates a new dataset `mnist_3569` with only the specified classes

## Training a Classifier Model
- Add model definition to `models`
- Change the model used in `run_classifier.py`
- Run `python3 run_classifier.py <dataset_name>`
- This saves the weights of the trained model in `weights/<dataset_name>` directory

## OOD Scores
- To add/modify a OOD score, add it as a function in `scores.py`

## Pretrained Scoring
- Change pretrained model path in `test_ood.py`
- Run `python3 test_ood.py -i <id_dataset_name> -o <ood_dataset_name> -s <score_name> -n <model_name> -m <model_metric> -e <model_epochs>`
- This saves scores for ID and OOD data as `.npy` files in `results/raw`

## OOD Finetuning
- Run `python3 finetune_ood.py  -i <id_dataset_name> -f <finetune_dataset_name> -n <model_name> -m <model_metric> -e <model_epochs>`
- This behaves like `run_classifier.py` and saves the model and the test predictions in the same way

## Finetuning Losses
- To add/modify a finetuning loss, add it as a `nn.Module` in `losses.py`

## Plots and Metrics for OOD Detection
- Run `python3 analyze_scores.py -i <id_dataset_name> -o <ood_dataset_name> -s <score_name>`
- This saves the density plots for the scores in `results/plots` 

## Running on Colab
- Create a Google Drive folder having the same root structure and upload the data
- Add a colab notebook to this folder, mount drive
- `cd` to created folder and run the `.py` scripts from notebook command line
- Sync to and from Drive when needed

## References
- [wetliu/energy_ood](https://github.com/wetliu/energy_ood)
- [tayden/ood-metrics](https://github.com/tayden/ood-metrics)
