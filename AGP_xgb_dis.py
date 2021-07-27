import utils
import pandas as pd
import numpy as np
import xgboost as xgb

# INPUT
metadata = 'metadata_gut_nr_noft.csv' # sys.argv[1]
otufile = 'OTU_TABLE_gut_nozero.csv' # sys.argv[2]

drop_cat = True

clade_level = 'asdf' # sys.argv[3]
bloom = True # sys.argv[4]
resample = False
thres = 1000
count = 100000

wanted_features = 'wanted_feat'

param = {'max_depth': 5, 'learning_rate': 0.1,
          'objective': 'binary:logistic', 'silent': True,
          'sample_type': 'uniform',
          'normalize_type': 'tree',
          'rate_drop': 0.4,
          'skip_drop': 0.5}
param['nthread'] = 6
param['eval_metric'] = 'auc'
num_round = 200

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
fo =open('xgbmodels/roc.txt','w')
#fo2 = open('fastaimodels/sn_sp.txt','w')
dis_list=list(disease_list.keys())
fo.write('Disease\tauc_val\t'+'\t'.join(dis_list)+'\n')
#fo2.write('Disease\tacc_val\t'+'\t'.join(dis_list)+'\n')
for dis in dis_list:
	cat_feats = utils.find_cat(list(DF_meta.columns))
	#df_dis,cat_feats = utils.prepare_disease_eq(dis,disease_list[dis],df,cat_feats,5)
	df_dis,cat_feats = utils.prepare_disease(dis,disease_list[dis],df,cat_feats)
	cat_dict = utils.cat_embsize(DF_meta,cat_feats)
	df_dis = df_dis.reset_index()
	df_dis.drop('index',axis=1,inplace=True)
	valid_idx = utils.testtrain_eq_age_corrected(df_dis,dis)
	
	if drop_cat:
		#================ DROP CATEGORICAL =============
		df_dis_cat = df_dis.copy()
		df_dis,cat_feats = utils.drop_cat(dis,df_dis,cat_feats)
	
		# =========================================== fastai ======
	df_cat = df_dis[cat_feats]
	feat,A_meta = utils.metadata_onehot(df_cat)
	#{{{{{{{{
#	A_num = df_dis[['bmi_corrected','age_years','elevation']].values
#	feat2 = cat_feats + ['bmi_corrected','age_years','elevation']
#	df_tax = df_dis[[f for f in df_dis.columns if f not in feat2 and f!=dis]]
#	attrib = feat + ['bmi_corrected','age_years','elevation']+list(df_tax.columns)
#	A = np.concatenate((A_meta,A_num,df_tax.values),axis=1)
	##

	feat2 = cat_feats + ['bmi_corrected','age_years','elevation']
	df_tax = df_dis[[f for f in df_dis.columns if f not in feat2 and f!=dis]]
	attrib = feat +list(df_tax.columns)
	A = np.concatenate((A_meta,df_tax.values),axis=1)
	
	t = np.zeros((A.shape[0]))
	t[df_dis[dis]=='Diagnosed by a medical professional (doctor, physician assistant)']=1
	t[df_dis[dis]=='T']=1
	A_val = A[valid_idx,:]
	t_val=t[valid_idx]
	tr_ind=[i for i in range(A.shape[0]) if i not in valid_idx]
	A_tr=A[tr_ind,:]
	t_tr=t[tr_ind]
	dtrain = xgb.DMatrix(A_tr, label=t_tr)
	evals = xgb.DMatrix(A_val, label=t_val)
	model = xgb.train(param, dtrain,evals=[(evals,'eval')],verbose_eval=False)#, early_stopping_rounds=20
	dtest = xgb.DMatrix(A_val)
	y = model.predict(dtest)
	auc,roc = utils.auc_calc(y,t_val,len(t_val)//2,len(t_val)//2)
	res = model.get_fscore()
	sec = [k for k in sorted(res, key=res.get, reverse=True)]
	fo.write(dis+'\t'+str(auc))
	for f in sec:
		fo.write('\t'+attrib[int(f[1:])])
	fo.write('\n')
	
fo.close()



# lrf=lea1rn.lr_find()
# y=np.array(learn.recorder.losses)
# x=learn.recorder.lrs
# plt.close('all')
# plt.plot(np.log10(x),y)
# plt.savefig('lr_slope.png')
