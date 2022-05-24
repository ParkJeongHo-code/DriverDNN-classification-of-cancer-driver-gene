
import tensorflow as tf
import argparse
import pandas as pd
import os
from sklearn.metrics import accuracy_score,roc_curve,auc
import numpy as np



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer_name', '--dis_name', help="TCGA cancer name", required=True)
    parser.add_argument('-test_data_directory', '--data_dir', help="", required=True)
    parser.add_argument('-result_folder_name', '--folder_name',default="result", help="", required=True)
    args = vars(parser.parse_args())
    return args

def preditc_data(inputs):
    os.mkdir(inputs['folder_name'])    
    driver_gene_list=list(pd.read_csv('using_Data/'+inputs['dis_name']+'_label.csv').loc[:,'gene_label'])
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
    test_data=pd.read_csv(inputs['data_dir'],sep=sep_)
    test_gene_list=test_data.loc[:,'gene_name']
    test_labels=[]
    for test_gene in list(test_gene_list):
        if test_gene in driver_gene_list:
            test_labels.append(1)
        else:
            test_labels.append(0)
    test_labels=list(np.array(test_labels).reshape(-1,))
    test_gene_list=pd.DataFrame(test_gene_list)
    test_data=test_data.loc[:,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]
    model1=tf.keras.models.load_model('trained_model/'+inputs['dis_name']+'_model_idx0.h5')
    model2=tf.keras.models.load_model('trained_model/'+inputs['dis_name']+'_model_idx1.h5')
    model3=tf.keras.models.load_model('trained_model/'+inputs['dis_name']+'_model_idx2.h5')
    predicts=model1.predict(test_data)
    pos_score=[]
    pred_label=[]
    for pred in predicts:
        pred_label.append(pred.argmax())
        pos_score.append(pred[1])
    fpr, tpr, thresholds = roc_curve(test_labels,pos_score,pos_label=1)
    test_auc_1=auc(fpr, tpr)
    test_gene_list['pos_score']=pos_score
    test_gene_list['classification']=pred_label
    test_gene_list.sort_values(by=['pos_score'], axis=0, ascending=False,inplace=True)
    test_gene_list.to_csv(inputs['folder_name']+'/result_1.csv',index=False)
    predicts=model2.predict(test_data)
    pos_score=[]
    pred_label=[]
    for pred in predicts:
        pred_label.append(pred.argmax())
        pos_score.append(pred[1])
    fpr, tpr, thresholds = roc_curve(test_labels,pos_score,pos_label=1)
    test_auc_2=auc(fpr, tpr)
    test_gene_list['pos score']=pos_score
    test_gene_list['classification']=pred_label
    test_gene_list.sort_values(by=['pos_score'], axis=0, ascending=False,inplace=True)
    test_gene_list.to_csv(inputs['folder_name']+'/result_2.csv',index=False)        
    predicts=model3.predict(test_data)
    pos_score=[]
    pred_label=[]
    for pred in predicts:
        pred_label.append(pred.argmax())
        pos_score.append(pred[1])
    fpr, tpr, thresholds = roc_curve(test_labels,pos_score,pos_label=1)
    test_auc_3=auc(fpr, tpr)
    test_gene_list['pos score']=pos_score
    test_gene_list['classification']=pred_label
    test_gene_list.sort_values(by=['pos_score'], axis=0, ascending=False,inplace=True)
    test_gene_list.to_csv(inputs['folder_name']+'/result_3.csv',index=False)  
    f = open(inputs['folder_name']+"/result_AUC.txt",'a')
    for i in [test_auc_1,test_auc_2,test_auc_3]:
        data = "dataset1 AUC: %f\n" % i
        f.write(data)
    f.close()

if __name__ == "__main__" :
    inputs = arg_parse()
    preditc_data(inputs)