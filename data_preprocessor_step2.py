#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:28:47 2021

@author: bmlserver
"""
from time import sleep 

import pandas as pd
from sklearn.metrics import balanced_accuracy_score,accuracy_score,roc_curve,auc,recall_score,precision_score,f1_score
from sklearn import metrics
import numpy as np
from pyensembl import EnsemblRelease
import re
import os
from multiprocessing import Process
import time
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras import layers,optimizers
from collections import Counter
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer_name', '--dis_name', help="TCGA cancer name", required=True)
    parser.add_argument('-step_1_data_dir', '--step_1_data', help="TCGA cancer name", required=True)
    parser.add_argument('-out_dir', '--save_dir', help="TCGA cancer name",default='', required=False)
    parser.add_argument('-exp_data_dir', '--exp_dir', help="TCGA cancer name",default='', required=False)
    args = vars(parser.parse_args())
    return args


def act(inputs):
    def cut_threshold(data,threshold):
        idxs=[idx for idx in range(data.shape[0]) if data.loc[idx,'LLS']>threshold]
        data=data.loc[idxs,:]
        return data
    def make_inter_dict(data):
        data2=pd.concat((data.loc[:,'entrez_id_1'].drop_duplicates(),data.loc[:,'entrez_id_2'].drop_duplicates()),axis=0)
        data2=data2.drop_duplicates()
        dict_={}
        for dis in list(data2):
            dict_[dis]=[]
        for gene1,gene2 in np.array(data.iloc[:,[1,2]]):
            dict_[gene1].append(gene2)
            dict_[gene2].append(gene1)
        return dict_
    dis_name=inputs['dis_name']# get disease name




    data_exp=pd.read_csv(inputs['exp_dir'],sep='\t')
    gene_exp=[]

    # delete version in ensembl name
    for i in range(data_exp.shape[0]):
        if '.' in data_exp.iloc[i,0]:
            gene_exp.append(data_exp.iloc[i,0].split('.')[0])
        else:
            gene_exp.append(data_exp.iloc[i,0])
    data_exp['Ensembl_ID']=gene_exp


    #%%
    data_exp=data_exp.set_index('Ensembl_ID')


    #%% filtering normal sample

    sample_id=list(data_exp.columns)
    new_id=[]
    good_sample_id=[]
    for i in range(len(sample_id)):

        if int(sample_id[i][-3:-1])>=10:
            good_sample_id.append(sample_id[i])
        else:
            new_id.append(sample_id[i])
            
    #%% get mean of normal samples gene expression
    good_samples=data_exp.loc[:,good_sample_id]
    good_sample_sum=good_samples.sum(axis=1)
    good_sample_sum=good_sample_sum/len(good_sample_id)

    bad_samples=data_exp.loc[:,new_id]
    bad_sample_sum=bad_samples.sum(axis=1)
    bad_sample_sum=bad_sample_sum/len(new_id)
    #%%
    deg__=bad_sample_sum-good_sample_sum
    if deg__.isna().sum()>=1:
        print('nan value in deg feature')
        quit()










    data=pd.read_csv(inputs['step_1_data'])
    ens_sym={}
    for i in range(data.shape[0]):
        ens_sym[data.loc[i,'gene_ens']]=data.loc[i,'gene_symbol']
    idx_=0
    col_l=list(data.columns)
    for col in list(data.columns):
        if col=='gene_ens':
            col_l[idx_]='gene'
        idx_+=1
    data.columns=col_l
    data.drop(['gene_symbol'],axis=1,inplace=True)
    #%%
    muta_gene=list(data.loc[:,'gene'].drop_duplicates())

    data.index=list(range(data.shape[0]))
    protein=pd.read_csv('/mnt/disk1/driver_gene/data/hsa_id_conv.txt',sep='\t')
    protein_filter=protein[protein['gene_biotype']=='protein_coding']
    pro_idx=[]
    iter_protein_filter=1
    total_iter=data.shape[0]
    for i in range(data.shape[0]):
        if data.loc[i,'gene'] in list(protein_filter.loc[:,'ensembl_gene_id']):
            pro_idx.append(i)
        if (iter_protein_filter/total_iter)*100 != 100:
            per=str((iter_protein_filter/total_iter)*100)
            if  per.index('.')==1:
                print(per[:3] + '% filtering protein coding gene processed..',end='\r',flush=True)
            else:
                print(per[:4] + '% filtering protein coding gene processed..',end='\r',flush=True)
        else:
            print(str((iter_protein_filter/total_iter)*100)[:5] + '% filtering protein coding gene processed..')
        iter_protein_filter+=1
    data=data.loc[pro_idx,:]
    data.index=list(range(data.shape[0]))






    ss=data.groupby('gene')
    total_gene=len(list(set(list(data.loc[:,'gene']))))
    #%% calculate gene mutation features(fraction)
    last_data=[]
    for_col=0
    esang=0
    iter=1
    for i,k in enumerate(ss):

        data_=k[1].iloc[:,2:]
        if for_col==0:# for make new columns
            cols=list(data_.columns)
            cols.insert(0,'gene')
            cols.append('muta_count')
        

            for_col+=1
        sum_=data_.sum(axis=0)

        sdaasd=0
        for u in sum_:
            if u%1 !=0:
                sdaasd+=1

        div=sum_.sum(axis=0)
        if div>0:
            sum_=sum_/div
        else:
            sum_=sum_/1


        if sum_.sum(axis=0)<0.9 or sum_.sum(axis=0)>1.1 :
            esang+=1
        

        
        sum_=list(sum_)
        sum_.insert(0,k[0])

        filter_k=k[1][(k[1]['stop_gained']>=1) | (k[1]['missense_variant']>=1) | (k[1]['frameshift_variant']>=1) | (k[1]['splice_region_variant']>=1)]                                          
        sum_.append(filter_k.shape[0])

        last_data.append(sum_)
        per=str((iter/total_gene)*100)
        if (iter/total_gene)*100 != 100:
            if  per.index('.')==1:

                print(per[:3] + '% mutation feature processed..',end='\r',flush=True)
            else:
                print(per[:4] + '% mutation feature processed..',end='\r',flush=True)
    
        else:
            print(per[:5] + '% mutation feature processed..')

        iter+=1
    last_data=pd.DataFrame(last_data)
    last_data.columns=cols

    last_data['deg']=list(deg__.loc[list(last_data.loc[:,'gene']),])
    last_data.dropna(subset=['deg'],inplace=True)
    print(last_data.isna().sum())
    #%%
    #%% make tool for label
    
    last_data_protein=last_data
    last_data_protein.index==list(range(last_data_protein.shape[0]))

        

    #%% filtering protein coding gene

    dis_pathway={}
    dis_pathway['BRCA']=['hsa05224','hsa04151','hsa04310','hsa04330','hsa04915','hsa04115','hsa03440', 'hsa04151','hsa04110', 'hsa04010']
    dis_pathway['PRAD']=['hsa05215','hsa00140','hsa04010','hsa04060','hsa04110','hsa04115','hsa04151','hsa04210','hsa05202']
    dis_pathway['BLCA']=['hsa05219','hsa04010','hsa04012','hsa04110','hsa04115','hsa04370','hsa04520']
    dis_pathway['PAAD']=['hsa05212','hsa04010','hsa04012','hsa04110','hsa04115','hsa04151','hsa04210','hsa04350','hsa04370','hsa04630']
    dis_pathway['SKCM']=['hsa05218','hsa04010','hsa04110','hsa04115','hsa04151','hsa04520','hsa04916']
    dis_pathway['COAD']=['hsa05210','hsa04010','hsa04012','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04350']
    dis_pathway['GBM']=['hsa05214','hsa04010','hsa04012','hsa04020','hsa04060','hsa04110','hsa04115','hsa04150']
    dis_pathway['SARC']=['hsa05200','hsa03320','hsa04010','hsa04020','hsa04024','hsa04060','hsa04066','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04330','hsa04340','hsa04350','hsa04370','hsa04510','hsa04512','hsa04520','hsa04630','hsa04915']
    dis_pathway['READ']=['hsa05210','has04010','hsa04012','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04350']
    dis_pathway['KIRC']=['hsa05211','hsa00020','hsa04010','hsa04066','hsa04120','hsa04350','hsa04370']
    dis_pathway['KIRP']=['hsa05211','hsa00020','hsa04010','hsa04066','hsa04120','hsa04350','hsa04370']
    dis_pathway['LUSC']=['hsa05223','hsa04010','hsa04012','hsa04014','hsa04020','hsa04110','hsa04115','hsa04151']
    dis_pathway['LUAD']=['hsa05223','hsa04010','hsa04012','hsa04014','hsa04020','hsa04110','hsa04115','hsa04151']
    dis_pathway['HNSC']=['hsa05200','hsa03320','hsa04010','hsa04020','hsa04024','hsa04060','hsa04066','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04330','hsa04340','hsa04350','hsa04370','hsa04510','hsa04512','hsa04520','hsa04630','hsa04915']
    dis_pathway_list=[]
    for dis in list(dis_pathway.keys()):
        dis_pathway_list.extend(dis_pathway[dis])
    dis_pathway_list=list(set(dis_pathway_list))




    pathway=pd.read_csv('/mnt/disk1/driver_gene/data/pathway/pathway_ensembl.csv')
    pathway2=pathway.drop('Unnamed: 0',axis=1)
    pathways=pathway2.loc[:,'pathway'].drop_duplicates()
    pathways.index=list(range(pathways.shape[0]))
    pathway_dict={}
    #각 pathway별로 gene 정리
    pathway_iter=1
    for jj in range(pathways.shape[0]):
        if pathways.loc[jj] in dis_pathway_list:
            gene_list=[pathway2.loc[i,'gene_id'] for i in range(pathway2.shape[0]) if pathway2.loc[i,'pathway']==pathways.loc[jj]]
            pathway_dict[pathways.loc[jj]]=gene_list
        per=str((pathway_iter/pathways.shape[0])*100)
        if (pathway_iter/pathways.shape[0])*100 != 100:
            if  per.index('.')==1:

                print(per[:3] + '% pathway list make',end='\r',flush=True)
            else:
                print(per[:4] + '% pathway list make',end='\r',flush=True)
    
        else:
            print(per[:5] + '% pathway list make')

        pathway_iter+=1

    add_col=[]
    add_col_dir_=[]
    pathway_iter=1
    for idx in range(last_data_protein.shape[0]):
        count=0
        count_dir=0
        important_pathway=0
        pathway_idx=0
        for pathway in dis_pathway[dis_name]:
            if pathway in list(pathway_dict.keys()):
                if last_data_protein.loc[idx,'gene'] in pathway_dict[pathway]:
                    if pathway_idx==0:
                        count_dir=1
                    else:
                        count+=1

                pathway_idx+=1
                    
        add_col_dir_.append(count_dir)
        add_col.append(count)
        per=str((pathway_iter/last_data_protein.shape[0])*100)
        if (pathway_iter/last_data_protein.shape[0])*100 != 100:
            if  per.index('.')==1:

                print(per[:3] + '% pathway feature make',end='\r',flush=True)
            else:
                print(per[:4] + '% pathway feature make',end='\r',flush=True)
    
        else:
            print(per[:5] + '% pathway feature make')

        pathway_iter+=1

    div_fact=len(dis_pathway[dis_name])-1
    last_data_protein['related_pathway']=np.array(add_col)
    last_data_protein['dir_pathway']=add_col_dir_
    train_data=last_data_protein
    splice=np.array(train_data.loc[:,'splice_region_variant'])
    inframe=np.array(train_data.loc[:,'inframe_insertion'])+np.array(train_data.loc[:,'conservative_inframe_deletion'])+np.array(train_data.loc[:,'inframe_deletion'])
    lost=np.array(train_data.loc[:,'start_lost'])+np.array(train_data.loc[:,'stop_lost'])
    col=['gene','synonymous_variant','stop_gained','missense_variant','frameshift_variant']
    #%%
    miss_ratio=[]
    for i in range(train_data.shape[0]):
        if train_data.loc[i,'synonymous_variant']+train_data.loc[i,'missense_variant'] != 0:
            miss_ratio.append(train_data.loc[i,'missense_variant']/(train_data.loc[i,'synonymous_variant']+train_data.loc[i,'missense_variant']))
        else:
            miss_ratio.append(0)

    #%%
    new_train=train_data.loc[:,col]
    new_train['splice']=splice
    new_train['inframe']=inframe
    new_train['lost_stop and start']=lost
    new_train['deg']=train_data.loc[:,'deg']
    new_train['related_pathway']=train_data.loc[:,'related_pathway']
    new_train['dir_pathway']=train_data.loc[:,'dir_pathway']
    new_train['muta_count']=train_data.loc[:,'muta_count']
    new_train['miss_ratio']=miss_ratio

    #%%
    col_l_2=list(new_train.columns)
    idxss=0
    new_train['gene_ens']=new_train['gene']
    for col in col_l_2:
        if col=='gene':
            col_l_2[idxss]='gene_ens'
    gene_syms=[]
    for gene in list(new_train.loc[:,'gene']):
        gene_syms.append(ens_sym[gene])
    new_train['gene_symbol']=gene_syms
    col_l_2.insert(1,'gene_symbol')
    new_train=new_train.loc[:,col_l_2]
    new_train.to_csv(inputs['save_dir']+dis_name+'_for_git_middle_step.csv')

    print('data_save')
    print(new_train)


    grap_data=pd.read_csv('/mnt/disk1/driver_gene/data/graph_data/humannet/humannetv3_ens_ppi.csv')
    grap_data=cut_threshold(grap_data,3.211)


    grap1_=list(grap_data.loc[:,'entrez_id_1'])
    grap2_=list(grap_data.loc[:,'entrez_id_2'])
    grap1_.extend(grap2_)
    grap1_=list(set(grap1_))
    print(len(grap1_))

    ad_node=make_inter_dict(grap_data)

    last_data_protein=new_train
    last_data_protein=last_data_protein.set_index('gene_ens')
    cancer_type_muta=list(last_data_protein.columns)
    cancer_type_muta=['stop_gained','splice','frameshift_variant','missense_variant']
    print(cancer_type_muta)
    graph_feat=[]
    max_=last_data_protein.loc[:,'muta_count']
    max_ = int(max_.quantile(.75))
    for_com_per=1
    under_t=len(list(last_data_protein.index))
    for g_name in list(last_data_protein.index):
        if last_data_protein.loc[g_name,cancer_type_muta].sum(axis=0)>0 and g_name in list(ad_node.keys()):
            count=0
            for adj_node in ad_node[g_name]:
                if adj_node in list(last_data_protein.index):
 
                    if last_data_protein.loc[adj_node,'muta_count']>max_ :
                        count+=1
            graph_feat.append(count)
        else:
            graph_feat.append(0)

        per=str((for_com_per/under_t)*100)
        if (for_com_per/under_t)*100 != 100:
            if  per.index('.')==1:

                print(per[:3] + '% pathway feature make',end='\r',flush=True)
            else:
                print(per[:4] + '% pathway feature make',end='\r',flush=True)
    
        else:
            print(per[:5] + '% pathway feature make')
        for_com_per+=1
    last_data_protein['PPI']=graph_feat

    path_std=last_data_protein.loc[:,'related_pathway'].std()
    path_mean=last_data_protein.loc[:,'related_pathway'].mean()
    path_up=last_data_protein.loc[:,'related_pathway']-path_mean
    last_data_protein['related_pathway']=list(path_up/path_std)

    muta_std=last_data_protein.loc[:,'muta_count'].std()
    muta_mean=last_data_protein.loc[:,'muta_count'].mean()
    muta_up=last_data_protein.loc[:,'muta_count']-muta_mean
    last_data_protein['muta_count']=list(muta_up/muta_std)
    

    ppi_std=last_data_protein.loc[:,'PPI'].std()
    ppi_mean=last_data_protein.loc[:,'PPI'].mean()
    ppi_up=last_data_protein.loc[:,'PPI']-ppi_mean
    last_data_protein['PPI']=list(ppi_up/ppi_std)
    print(last_data_protein)
    print(last_data_protein.shape)

    last_data_protein.to_csv(inputs['save_dir']+dis_name+'_input_data.csv')
        






if __name__ == "__main__" :
    inputs = arg_parse()
    act(inputs)
