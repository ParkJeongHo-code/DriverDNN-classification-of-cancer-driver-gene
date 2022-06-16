# DriverDNN: classification-of-cancer-driver-gene
Goal of this project is classification of specific cancer driver gene.
In our study, we classify BRCA, PAAD and BLCA cancer driver gene by deep neural network.

# training by DriverDNN
if you want to train your data, you can use DriverDNN_training.py
## argument
### 1.cancer_name
cancer_name argument is meaning TCGA cancer name.(One of the following: BRCA, PAAD, PRAD)

### 2.data_directory 
when you have one dataset and you want to seperate the dataset by train and test,don't use test_data_directory argument, and use only data_directory argument. 
we divide train,test dataset by 3 fold stratified cross validation. So we make three dataset. by using 3 fold stratified cross validation, you can get three model in one raw dataset.
### example 
    python DriverDNN_training.py -cancer_name BRCA -data_directory sample_data/TCGA_BRCA_input.csv -lr 0.001 -epoch 10 -batch 64 -early_stop 5 -result_folder_name result 

### 3.lr
lr is meaning learning rate of model training.

### 4.epoch
epoch is meaning epoch of model training.

### 5.batch
batch argument is meaning batch size of model training.

### 6.early_stop
early_stop argument is number of epochs with no improvement after which training will be stopped.

### 7.result_folder_name
result_folder_name argument is name of folder that save result and model.

### 8.test_data_directory
when you have train dataset and test dataset separately, you can use data_directory argument for train dataset and use test_data_directory for test dataset.
### example 
    python DriverDNN_training.py -cancer_name BRCA -data_directory sample_data/TCGA_BRCA_input.csv -lr 0.001 -epoch 10 -batch 64 -early_stop 5 -result_folder_name result -test_data_directory sample_data/CPTAC_BRCA_input.csv

# predict by DriverDNN model which is already trained
if you want to predict your data by our trained model, you can use DriverDNN_predict.py
### example 
    python DriverDNN_predict.py -cancer_name BRCA -test_data_directory sample_data/TCGA_BRCA_input.csv -result_folder_name result_1

## argument
### 1.cancer_name
cancer_name argument is meaning TCGA cancer name.

### 2.test_data_directory
directory of data that you wanna predict.

### 3.result_folder_name
folder name that you wanna save result file.

# reproducing an experiment in paper
if you want to reproduce our experiment in paper , you can use reproduce_experiment.py
## argument
argument is same with DriverDNN_training.py
## cross validation based 
you can reproduce cross validation based experiment by command below.
### command for reproducing cross validation based experiment
    python reproduce_experiment.py -cancer_name BRCA -data_directory sample_data/TCGA_BRCA_input.csv -lr 0.001 -epoch 100 -batch 64 -early_stop 10 -result_folder_name result 
## The experiment that using CPTAC dataset 
you can reproduce experiment that using CPTAC dataset by command below.
### command for reproducing experiment that using CPTAC dataset 
    python reproduce_experiment.py -cancer_name BRCA -data_directory sample_data/TCGA_BRCA_input.csv -lr 0.001 -epoch 100 -batch 64 -early_stop 10 -result_folder_name result -test_data_directory sample_data/CPTAC_BRCA_input.csv

