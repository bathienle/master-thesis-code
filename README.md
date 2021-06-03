# Code for the Master Thesis

## Requirements

The implementation is done in Python 3.8 and it requires the following packages:

- numpy >= 1.20.1
- scipy >= 1.6.1
- matplotlib >= 3.3.4
- pytorch >= 1.8.0
- torchvision >= 0.9.0
- pillow >= 8.1.0
- scikit-image >= 0.18.1
- pandas >= 1.2.2
- opencv >= 4.5.1
- Cytomine

### Conda environment

Two conda environments are provided, namely *thesis* and *test*.
The former is to use for the training, evaluation, and the experiments. The latter is used for the testing.

* To create the *thesis* environment:
```
conda env create --file env/environment.yml
```
* To create the *test* environment:
```
conda env create --file env/test.yml
```

## Usage

### Download the datasets from Cytomine

To download the dataset from Cytomine:
```
cd src/data/
conda activate thesis
python3 get_data.py --host hostname --public_key public_key --private_key private_key --project_id project_id --term term --path path_to_store --users "username1 username2"
```
where *hostname* is the name of the Cytomine host, *public_key* and *private_key* is the public and private keys of your Cytomine account, *project_id* is the id of the project to download the images from, *term* is the specific annotation terms to download, e.g., "Bronchus", *path* is the path to store the dataset, and an optional argument *users* to download the annotations from specific users only. If not mentionned, it will download all the annotations.

### Training

To train the neural network:
```
cd src/
conda activate thesis
python3 train.py --epochs epoch --type type --data path_to_data --dest dest_path --stat path_to_csv.csv
```
where *epoch* is the number of epochs, *type* is the type of object to segment, *path_to_data* is the path to the dataset, *dest_path* is the path to save the trained model, and *path_to_csv* the path to the CSV file to write the loss during training.

To have the complete list of parameters:
```
python3 train.py -h
```

Since this implementation relies on the PyTorch framework, it will automatically use the GPU if it is enabled.

### Evaluation

To evaluate the trained neural network:
```
cd src/
conda activate thesis
python3 evaluate.py --data path_to_data --stat path_to_csv --weight path_to_weight --type type
```
where *path_to_data* is the path to the dataset and *path_to_csv* the path to store the CSV file, *path_to_weight* is the path to the weight with extension *.pth, and *type* is the type of object to segment. A remark about the *stat* option, in this case there is no need to specify the name of the csv as for the training. It will be automatically saved in a CSV file named *type*-evaluation.csv, where *type* is the type giving in the command line argument.

### Testing

To test the program for one image:
```
cd src/test/
conda activate test
python3 visualize.py --weight path_to_weight --dest path_to_dest
```
where *path_to_weight* is the path to the weight with extension *.pth and *path_to_dest* is the path to store the resulting segmentation mask.
