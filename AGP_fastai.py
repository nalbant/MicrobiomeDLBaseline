import utils
import sys
import pandas as pd
import numpy as np
from fastai import *
from fastai.tabular import *

# INPUT
metadata = 'metadata_gut_nr_noft.csv' # sys.argv[1]
otufile = 'OTU_TABLE_gut_nozero.csv' # sys.argv[2]

clade_level = 's' # sys.argv[3]
bloom = True # sys.argv[4]
resample = False
thres = 1000
count = 100000

wanted_features = 'wanted_feat'
# DATA FILES
DF_meta = pd.read_csv(metadata,index_col='#SampleID')
DF = pd.read_csv(otufile)

# =========================================== REDUCE SEQUENCE DATA ======
#remove bloom
if bloom:
	DF = utils.remove_bloom(DF)

#otu collapse
A,idrows,taxacols = utils.otu_collapse(DF,clade_level)

#filter samples with few reads
A,idrows = utils.filter_undersampled(A,idrows,thres)

#resample otus
if resample:
	A=utils.otu_resampler(A,count)
else: #else normalize
	A = A/A.sum(axis=1)[:,None]

DF = pd.DataFrame(A,columns = taxacols, index = idrows)
idx=DF.index

#filter metadata with no sequences
DF_meta = utils.filter_nonsequenced_samples(idx,DF_meta)

# =========================================== REDUCE METADATA ======

#pick selected features
DF_meta,feats = utils.wanted_metadata(DF_meta, wanted_features)


#remove features with  unbalances class distribution
DF_meta = utils.remove_unbalanced_features(DF_meta)
for f in ['bmi_corrected','age_years','elevation']:
	DF_meta = utils.filter_cat(DF_meta,f)

DF_meta['bmi_corrected']=pd.to_numeric(DF_meta['bmi_corrected'])
DF_meta['age_years']=pd.to_numeric(DF_meta['age_years'])
DF_meta['elevation']=pd.to_numeric(DF_meta['elevation'])
#return categorical features

# =========================================== DISEASE ======

disease_list_file = 'disease'
disease_list={}
for line in open(disease_list_file):
	dl = line.strip().split('\t')
	disease_list[dl[0]]=dl[1:]

df =  pd.concat([DF_meta, DF], axis=1)
df = df.reset_index()
df.drop('index',axis=1,inplace=True)

##################3
fo =open('fastaimodels/roc.txt','w')
fo1 =open('fastaimodels/acc.txt','w')
#fo2 = open('fastaimodels/sn_sp.txt','w')
dis_list=list(disease_list.keys())
fo.write('Disease\tauc_val\t'+'\t'.join(dis_list)+'\n')
fo1.write('Disease\tacc_val\t'+'\t'.join(dis_list)+'\n')
#fo2.write('Disease\tacc_val\t'+'\t'.join(dis_list)+'\n')
for dis in dis_list:
	fo.write(dis+'\n')
	fo1.write(dis+'\n')
	cat_feats = utils.find_cat(list(DF_meta.columns))
	#df_dis,cat_feats = utils.prepare_disease_eq(dis,disease_list[dis],df,cat_feats,5)
	df_dis,cat_feats = utils.prepare_disease(dis,disease_list[dis],df,cat_feats)
	cat_dict = utils.cat_embsize(DF_meta,cat_feats)
	df_dis = df_dis.reset_index()
	df_dis.drop('index',axis=1,inplace=True)
	valid_idx = utils.testtrain_eq_age_corrected(df_dis,dis)

	#================ DROP CATEGORICAL =============
	df_dis_cat = df_dis.copy()
	#df_dis,cat_feats = utils.drop_cat(dis,df_dis,cat_feats)

	# =========================================== fastai ======
	if valid_idx != -1:
		dep_var = dis
		procs = [FillMissing, Categorify]
		data = TabularDataBunch.from_df('fastaimodels', df_dis, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_feats)
		learn = tabular_learner(data, layers=[500,500],ps = 0.5, metrics=accuracy)
		auc_500_2,acc_500_2,roc_500_2,sn_500_2,sp_500_2 = utils.fastai_acc(learn,data,len(valid_idx),1e-2)
		learn = tabular_learner(data, layers=[500,500],ps = 0.5, metrics=accuracy)
		auc_500_1,acc_500_1,roc_500_1,sn_500_1,sp_500_1 = utils.fastai_acc(learn,data,len(valid_idx),1e-1)
		learn = tabular_learner(data, layers=[500,50],ps = 0.5, metrics=accuracy)
		auc_50_2,acc_50_2,roc_50_2,sn_50_2,sp_50_2 = utils.fastai_acc(learn,data,len(valid_idx),1e-2)
		learn = tabular_learner(data, layers=[50,50],ps = 0.05, metrics=accuracy)
		auc_50_1,acc_50_1,roc_50_1,sn_50_1,sp_50_1 = utils.fastai_acc(learn,data,len(valid_idx),1e-2)
		learn = tabular_learner(data, layers=[500,50,5],ps = 0.5, metrics=accuracy)
		auc_555,acc_555,roc_555,sn_555,sp_555 = utils.fastai_acc(learn,data,len(valid_idx),1e-2)

		au =np.array([auc_500_2,auc_500_1,auc_50_2,auc_50_1,auc_555])
		ac =np.array([acc_500_2,acc_500_1,acc_50_2,acc_50_1,acc_555])
		ro =np.array([roc_500_2,roc_500_1,roc_50_2,roc_50_1,roc_555])
		sns =np.array([sn_500_2,sn_500_1,sn_50_2,sn_50_1,sn_555])
		sps =np.array([sp_500_2,sp_500_1,sp_50_2,sp_50_1,sp_555])
		ia = au.argmax()
		auc=au[ia]
		acc=ac[ia]
		roc=ro[ia]
		sn=sns[ia]
		sp=sps[ia]
		fo.write(dis+'\t'+str(auc)+'\t')
		fo1.write(dis+'\t'+str(acc)+'\t')
		#fo2.write(dis+'\t'+str(df)+'\t')
		aucc,accur = utils.disease_transfer_eq(dis_list,df_dis_cat,roc,valid_idx)
		fo.write('\t'.join([str(u) for u in aucc])+'\n')
		fo1.write('\t'.join([str(u) for u in accur])+'\n')
fo.close()
fo1.close()



# lrf=lea1rn.lr_find()
# y=np.array(learn.recorder.losses)
# x=learn.recorder.lrs
# plt.close('all')
# plt.plot(np.log10(x),y)
# plt.savefig('lr_slope.png')
