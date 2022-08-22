

import models
import tensorflow as tf
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from collections import Counter
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer_name', '--dis_name', help="TCGA cancer name", required=True)
    parser.add_argument('-data_directory', '--data_dir', help="", required=True)
    parser.add_argument('-lr', '--learning_rate', help="", required=True)
    parser.add_argument('-epoch', '--epochs', help="", required=True)
    parser.add_argument('-batch_size', '--batch_size_', help="", required=True)
    parser.add_argument('-early_stop', '--early_stop_', help="", required=True)
    parser.add_argument('-result_folder_name', '--folder_name',default="result", help="", required=True)
    parser.add_argument('-test_data_directory', '--test_data_dir',default="NA", help="", required=False)
    args = vars(parser.parse_args())
    return args

def train_model(inputs):

    dis_name=inputs['dis_name']
    driver_gene_list=list(pd.read_csv('using_Data/'+dis_name+'_label.csv').loc[:,'gene_label'])
    file_d=list(inputs['data_dir'])
    file_d.reverse()
    reverse_train_data_dir=''.join(file_d) 
    point_loc=reverse_train_data_dir.find('.')
    reverse_extension=list(reverse_train_data_dir[:point_loc])
    reverse_extension.reverse()
    f_extension=''.join(reverse_extension)
    if f_extension=='tsv' or f_extension=='txt':
        sep_='\t'
    else:
        sep_=','
    train_data=pd.read_csv(inputs['data_dir'],sep=sep_)
    train_data=train_data.loc[:,['gene_name','synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]
    train_labels=[]
    for train_gene in list(train_data.loc[:,'gene_name']):
        if train_gene in driver_gene_list:
            train_labels.append(1)
        else:
            train_labels.append(0)
    train_data.drop('gene_name',inplace=True,axis=1)
    if inputs['test_data_dir']=='NA':   
        kf = StratifiedKFold(n_splits=3,shuffle=True)
        model_idx=1
        os.mkdir(inputs['folder_name'])
        for train_index, test_index in kf.split(np.array(train_data),np.array(train_labels)):
            train_x=np.array(train_data.loc[train_index,:])
            test_x=np.array(train_data.loc[test_index,:])
            train_lbls=np.array(train_labels)[train_index]
            test_lbls=np.array(train_labels)[test_index]

            if dis_name=='BRCA':
                train_model=models.model_BRCA((train_x.shape[1]))
            elif dis_name=='PAAD':
                train_model=models.model_PAAD((train_x.shape[1]))
            elif dis_name=='PRAD':
                train_model=models.model_PRAD((train_x.shape[1]))
            elif dis_name=='BLCA':
                train_model=models.model_BLCA((train_x.shape[1]))
            elif dis_name=='GBM':      
                train_model=models.model_GBM((train_x.shape[1]))
            elif dis_name=='READ':
                train_model=models.model_READ((train_x.shape[1]))
            elif dis_name=='SKCM':
                train_model=models.model_SKCM((train_x.shape[1]))
            elif dis_name=='KIRP':      
                train_model=models.model_KIRP((train_x.shape[1]))
            elif dis_name=='KIRC':      
                train_model=models.model_KIRC((train_x.shape[1]))
            elif dis_name=='LUSC':      
                train_model=models.model_LUSC((train_x.shape[1]))
            elif dis_name=='LUAD':      
                train_model=models.model_LUAD((train_x.shape[1]))
            elif dis_name=='SARC':      
                train_model=models.model_SARC((train_x.shape[1]))
            elif dis_name=='COAD':      
                train_model=models.model_COAD((train_x.shape[1]))
            elif dis_name=='HNSC':      
                train_model=models.model_HNSC((train_x.shape[1]))
                
            count_label=Counter(list(train_lbls))
            pos_=count_label[1]
            neg_=count_label[0]


            total=pos_+neg_
            weight_for_0 = (1 / neg_) * (total / 2.0)
            weight_for_1 = (1 / pos_) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}
            epochs=int(inputs['epochs'])
            batch_size=int(inputs['batch_size_'])
            early_stop=int(inputs['early_stop_'])
            learning_rate=float(inputs['learning_rate'])

            class_w=class_weight
            save_model_dir=inputs['folder_name']+'/model_test'+str(model_idx)+'.h5'
            models.training(train_model,train_x,train_lbls,test_x,test_lbls,epochs,batch_size,early_stop,class_w,save_model_dir,learning_rate)
            model_idx+=1
    else:
        train_x=np.array(train_data)
        train_lbls=np.array(train_labels)
        test_file_d=list(inputs['test_data_dir'])
        test_file_d.reverse()
        reverse_test_data_dir=''.join(test_file_d) 
        test_point_loc=reverse_test_data_dir.find('.')
        test_reverse_extension=list(reverse_test_data_dir[:test_point_loc])
        test_reverse_extension.reverse()
        test_f_extension=''.join(test_reverse_extension)
        if test_f_extension=='tsv' or test_f_extension=='txt':
            test_sep_='\t'
        else:
            test_sep_=','
        test_data=pd.read_csv(inputs['test_data_dir'],sep=test_sep_)
        test_data=test_data.loc[:,['gene_name','synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]
        test_labels=[]
        for test_gene in list(test_data.loc[:,'gene_name']):
            if test_gene in driver_gene_list:
                test_labels.append(1)
            else:
                test_labels.append(0)
        test_data.drop('gene_name',inplace=True,axis=1)
        test_x=np.array(test_data)
        test_lbls=np.array(test_labels)
        if dis_name=='BRCA':
            train_model=models.model_BRCA((train_x.shape[1]))
        elif dis_name=='PAAD':
            train_model=models.model_PAAD((train_x.shape[1]))
        elif dis_name=='PRAD':
            train_model=models.model_PRAD((train_x.shape[1]))
        elif dis_name=='BLCA':
            train_model=models.model_BLCA((train_x.shape[1]))
        elif dis_name=='GBM':      
            train_model=models.model_GBM((train_x.shape[1]))
        elif dis_name=='READ':
            train_model=models.model_READ((train_x.shape[1]))
        elif dis_name=='SKCM':
            train_model=models.model_SKCM((train_x.shape[1]))
        elif dis_name=='KIRP':      
            train_model=models.model_KIRP((train_x.shape[1]))
        elif dis_name=='KIRC':      
            train_model=models.model_KIRC((train_x.shape[1]))
        elif dis_name=='LUSC':      
            train_model=models.model_LUSC((train_x.shape[1]))
        elif dis_name=='LUAD':      
            train_model=models.model_LUAD((train_x.shape[1]))
        elif dis_name=='SARC':      
            train_model=models.model_SARC((train_x.shape[1]))
        elif dis_name=='COAD':      
            train_model=models.model_COAD((train_x.shape[1]))
        elif dis_name=='HNSC':      
            train_model=models.model_HNSC((train_x.shape[1]))
        count_label=Counter(list(train_lbls))
        pos_=count_label[1]
        neg_=count_label[0]


        total=pos_+neg_
        weight_for_0 = (1 / neg_) * (total / 2.0)
        weight_for_1 = (1 / pos_) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        epochs=int(inputs['epochs'])
        batch_size=int(inputs['batch_size_'])
        early_stop=int(inputs['early_stop_'])
        learning_rate=float(inputs['learning_rate'])
        class_w=class_weight
        save_model_dir=inputs['folder_name']+'/model_test_full.h5'
        models.training(train_model,train_x,train_lbls,test_x,test_lbls,epochs,batch_size,early_stop,class_w,save_model_dir,learning_rate)
if __name__ == "__main__" :
    inputs = arg_parse()
    train_model(inputs)


        


