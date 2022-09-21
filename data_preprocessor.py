import pandas as pd
import numpy as np
from pyensembl import EnsemblRelease
import re
import os
from multiprocessing import Process
import time
import sys
import json
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer_name', '--dis_name', help="TCGA cancer name", required=True)
    parser.add_argument('-out_dir', '--save_dir', help="directory for save file",default='', required=False)
    parser.add_argument('-exp_data_dir', '--exp_dir', help="gene expression data directory",default='', required=False)
    parser.add_argument('-muta_data_dir', '--muta_dir', help="maf file directory",default='', required=False)

    args = vars(parser.parse_args())
    return args

class make_input():
    def __init__(self):
        self.data=EnsemblRelease(102)
        
    def gene_to_ensembl(self,data,re_dict):
        print('gene ensembl mapping')
        data.index=list(range(data.shape[0]))
        ensembl=[]
        for i  in range(data.shape[0]):
            try:
                #print(self.data.gene_ids_of_gene_name(data.loc[i,'gene'])[0])
                ensembl.append(self.data.gene_ids_of_gene_name(data.loc[i,'gene_symbol'])[0])
            except ValueError:
                if data.loc[i,'gene_symbol'] in list(re_dict.keys()):
                    ensembl.append(re_dict[data.loc[i,'gene_symbol']])
                else:
                    ensembl.append(data.loc[i,'gene_symbol'])

                    
          
        data['gene_ens']=ensembl
        print('gene ensembl mapping end')

        return data
        
    def get_location(self,data,ref_data):
        starts=[]
        ends=[]
        for i in range(data.shape[0]):
            try:
                gene_name=self.data.genes_by_name(data.iloc[i,1])[0]
                starts.append(gene_name.start)
                ends.append(gene_name.end) 
            except ValueError:
                idx=[j for j in range(ref_data.shape[0]) if data.loc[i,'gene'] in str(ref_data.loc[j,'prev_symbol'])]
                if len(idx) != 0:
                    try:
                        new_symbol=ref_data.loc[idx[0],'symbol']
                        gene_name=self.data.genes_by_name(new_symbol)[0]
                        starts.append(gene_name.start)
                        ends.append(gene_name.end) 
                    except ValueError:
                        starts.append(np.nan)
                        ends.append(np.nan)    
                else:
                    starts.append(np.nan)
                    ends.append(np.nan)
                    

        data['gene_start']=starts
        data['gene_end']=ends
        return data
    def make_snp_indel_col(self,concat_data):
        print('make snp indel')
        classi_snp_indel=concat_data.loc[:,['ref','alt']]
        snp_or_indel=[]
        for i in range(classi_snp_indel.shape[0]):
            if '-' in list(classi_snp_indel.iloc[i,:]):
                snp_or_indel.append('indel')
            else:
                snp_or_indel.append('snp')
        concat_data['snp_or_indel']=snp_or_indel
        return concat_data
        
    def remake_eff(self,data):
        print('remake_eff')
        data.index=list(range(data.shape[0]))
    
        for i in range(data.shape[0]):
            if data.loc[i,'snp_or_indel'] =='snp':
                data.loc[i,'effect']='snp_'+data.loc[i,'effect']
            else:
                data.loc[i,'effect']='indel_'+data.loc[i,'effect']
    
        return data
        
    
    def make_last_data(self,concat_data,ref_data2):
        print('count mutation type')
    
        data_ref2=pd.read_csv(ref_data2,sep='\t',low_memory=False)
        so_varient=data_ref2.loc[:,'SO term']
        so_varient=[re.sub(" ","_",i) for i in so_varient] 
        so_varient=list(set(so_varient))
        so_varient.extend(['non_coding_transcript_variant'])



        last_dict=[]
        for key ,gene_group in concat_data.groupby(['Sample_ID','gene_ens']):
            gene_var_feat=list(np.zeros((len(so_varient))))     
            var_eff=gene_group.loc[:,'effect']         
            for eff in var_eff:
                idxxx=[idx for idx in range(len(so_varient)) if eff.split(',')[1] in so_varient[idx]]
                gene_var_feat[idxxx[0]]+=1
            last_result=[key[0],key[1],gene_group.loc[list(gene_group.index)[0],'gene_symbol']]
            last_result.extend(gene_var_feat)
            last_dict.append(last_result)
            last_result=[]
        feature=['Sample_id','gene_ens','gene_symbol']
        feature.extend(so_varient)
        last_dict=np.array(last_dict)
        return_data=pd.DataFrame(last_dict,columns=feature)    
        print('count mutation type end')
        return return_data
    def remove(self,data):
        idx=[]
        removes=[]
        data.dropna(subset=['effect'],inplace=True)
        data.index=list(range(data.shape[0]))
        for i in range(data.shape[0]):
            if ';' in data.loc[i,'effect']:
                idx.append(i)
                h=0
                for eff in str(data.loc[i,'effect']).split(';'):
                    df=data.loc[i,:].copy()
                    df.loc['effect']=eff
                    removes.append(df)
    
                        
    
        return     pd.concat([data.drop(idx,axis=0),pd.concat(removes,axis=1).transpose()])
    def make_dict(self,ref_data):
        re={}
        for i in range(ref_data.shape[0]):
            if ref_data.loc[i,'ensembl_gene_id'] != None:
        
                for prev in str(ref_data.loc[i,'prev_symbol']).split('|'):
                        re[prev]=ref_data.loc[i,'ensembl_gene_id']
                        
                for rece in str(ref_data.loc[i,'symbol']).split('|'):
                        re[rece]=ref_data.loc[i,'ensembl_gene_id']
                    
   
        return re

    
    
    def concat_exp_muta(self,disease_name,base_dir_out,data_exp_file,data_muta_file,ref_data2='C:/Users/user/Desktop/Park/project/few_shot_project/data/for_ref_data/soterm_list.txt',ref_data="C:/Users/user/Desktop/Park/project/few_shot_project/data/for_ref_data/prev_gene_to_new.txt"):
        data_exp=pd.read_csv(data_exp_file,sep='\t',low_memory=False)


        data_muta=pd.read_table(data_muta_file,header=5)        
        col_n=['Tumor_Sample_Barcode','Hugo_Symbol','Chromosome','Start_Position','End_Position','Reference_Allele','Tumor_Seq_Allele2','HGVSp','all_effects','FILTER']
        data_muta=data_muta.loc[:,col_n]
        col_change={'Tumor_Sample_Barcode':'Sample_ID','Hugo_Symbol':'gene_symbol','Chromosome':'chrom','Start_Position':'start','End_Position':'end','Reference_Allele':'ref','Tumor_Seq_Allele2':'alt','HGVSp':'Amino_Acid_Change','all_effects':'effect','FILTER':'filter','':'dna_vaf'}
        new_col=[]
        for i in list(data_muta.columns):
            new_col.append(col_change[i])
        data_muta.columns=new_col
        new_sample_id=[]
        for s_id in list(data_muta.loc[:,'Sample_ID']):
            s_list=s_id.split('-')
            new_sample=''
            for iiiii in range(4):
                if iiiii==0:
                    new_sample+=s_list[iiiii]
                else:
                    new_sample=new_sample+'-'+s_list[iiiii]
            new_sample_id.append(new_sample)
        data_muta.loc[:,'Sample_ID']=new_sample_id
        gene_enbm=self.make_dict(ref_data)

        data_muta=self.gene_to_ensembl(data_muta, gene_enbm)
        data_muta=self.remove(data_muta)
        data_muta.index=list(range(data_muta.shape[0]))

        gene_exp=[]
        for i in range(data_exp.shape[0]):
            idx=[jj for jj in range(len(data_exp.iloc[i,0])) if data_exp.iloc[i,0][jj] =='.']
            if len(idx) != 0:
                gene_exp.append(data_exp.iloc[i,0][:idx[0]])
            else:

                gene_exp.append(data_exp.iloc[i,0]) 
        data_exp['Ensembl_ID']=gene_exp
        data_exp.set_index('Ensembl_ID',inplace=True)

        data_muta_last=data_muta
        data_muta_last.index=list(range(data_muta_last.shape[0]))
        data_muta_last_2=self.make_last_data(data_muta_last,ref_data2)


        print('end step 1')
        if base_dir_out[-1]=='/':
            data_muta_last_2.to_csv(base_dir_out+disease_name+"_exp_muta_concat.csv",index=False)
            step_data_dir=base_dir_out+disease_name+"_exp_muta_concat.csv"
        else:
            data_muta_last_2.to_csv(base_dir_out+'/'+disease_name+"_exp_muta_concat.csv",index=False)
            step_data_dir=base_dir_out+'/'+disease_name+"_exp_muta_concat.csv"
            base_dir_out=base_dir_out+'/'
        print('start data_preprocessor_step2.py')

        os.system('python3 data_preprocessor_step2.py -cancer_name '+disease_name+' -step_1_data_dir '+step_data_dir+' -out_dir '+base_dir_out+' -exp_data_dir '+data_exp_file)
 
    def make_every_data(self,disease_name,base_dir_out,data_muta_dir,data_exp_dir,ref_data11,ref_data22):
        ref_data11=pd.read_csv(ref_data11)
        all_combine=[[data_exp_dir, data_muta_dir]]
        if __name__=='__main__':
            procs=[]
            for i in range(len(all_combine)):
                proc=Process(target=self.concat_exp_muta,args=(disease_name,base_dir_out,all_combine[i][0],all_combine[i][1],ref_data22,ref_data11))
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
def make_data(inputs):
    
    ref_data11='./for_ref/gene_ens_map.csv'
    ref_data22='./for_ref/soterm_list.txt'
        
    make_input().make_every_data(inputs['dis_name'],inputs['save_dir'], inputs['muta_dir'], inputs['exp_dir'], ref_data11, ref_data22)



if __name__ == "__main__" :
    inputs = arg_parse()
    make_data(inputs)
    
    
