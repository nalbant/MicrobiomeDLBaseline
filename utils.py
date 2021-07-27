import json
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from scipy import stats
import time
#from fastai import *
#from fastai.tabular import *

#def otuID(file):
#	with open(file) as fi:
#		L = fi.readlines()
#	otu_ID={}
#	for l in L:
#		ll=l.strip().split('\tk__')
#		otu_ID[ll[0]+'.0']=ll[1]
#	return otu_ID

def remove_bloom(DF):
	with open('bloom/bloomlist') as fi:
		bl_id = fi.read().strip().split('\t')
	allotu = set(DF.columns)
	bl_exists=[]
	for it in bl_id:
		if it in allotu:
			bl_exists.append(it)
	DF.drop(bl_exists,axis=1,inplace=True)
	return DF

def otu_collapse(DF,level):
	if level=='non':
		A = DF.values[:,1:].astype('float')
		idrows = list(DF['#OTU ID'])
		taxacols = list(DF.columns[1:])
		return A,idrows,taxacols
	else:
		with open('OTU_ID.json', 'r') as fp:
			otu_ID = json.load(fp)
		allotu = list(DF.columns)[1:]
		taxa=[]
		ids=[]
		for o in allotu:
			taxa.append(otu_ID[o])
			ids.append(o)
		collapids = defaultdict(list)
		for i,t in enumerate(taxa):
			collapids[t.split('; '+level+'__')[0]].append(ids[i])
		h,w = DF.shape
		A = np.zeros((h,len(collapids)))	
		taxacols=[]
		for i,t in enumerate(collapids):
			taxacols.append(t)
			A[:,i] = DF[collapids[t]].sum(axis=1).values
		idrows = list(DF['#OTU ID'])
		return A,idrows,taxacols


def filter_undersampled(A,idrows,thres):
	idx = A.sum(axis=1)>thres
	return A[idx,:],[ii for i,ii in enumerate(idrows) if idx[i]]

def filter_nonsequenced_samples(idrows,DF_meta):
	return DF_meta.loc[idrows]

def otu_resampler(A,count):
	AA=np.zeros_like(A)
	h,w = A.shape
	for i in range(h):
		AA[i,:] = np.bincount(np.random.choice(w,count,p=A[i,:]/A[i,:].sum()),minlength=w)
	return AA

def filter_bmi(DF_meta):
	bmi=[]
	idx=[]
	bmi_act = list(DF_meta['bmi'])
	bmi_correct = list(DF_meta['bmi_corrected'])
	for i,bm in enumerate(bmi_act):
		inde=-1
		bodmas=0.0
		if bm.lower()!='not provided' and bm.lower()!='unspecified':
			if 15.0 < float(bm)<60.0:
				inde=i
				bodmas=float(bm)
		if bmi_correct[i].lower()!='not provided' and bmi_correct[i].lower()!='unspecified':
			if 15.0 < float(bmi_correct[i])<60.0:
				inde=i
				bodmas=float(bmi_correct[i])
		if bodmas>0.0:
			bmi.append(bodmas)
			idx.append(inde)
	return idx,bmi,DF_meta.iloc[idx]

def filter_cat(DF_meta,feat):
	idx = DF_meta[feat]=='Not provided'
	idp = DF_meta[feat]!='Not provided'
	DF_meta[feat][idx]=DF_meta[feat][idp].median()
	return DF_meta

def remove_otus_unwanted_metadata(idx,A,idrows):
	return A[idx], [idrows[i] for i in idx]

def wanted_metadata(DF_meta,mtfile):
	#with open('meta_features') as fi:
	with open(mtfile) as fi:
		feat = fi.read().split('\n')
	return DF_meta[feat],feat

def remove_unbalanced_features(DF_meta):
	remove_list=[]
	a=list(DF_meta.columns)
	for aa in a:
		b = DF_meta[aa].value_counts()
		if b.sum()-b.max()<5:
			remove_list.append(aa)
	return DF_meta.drop(remove_list,axis=1)

def metadata_onehot(DF_meta):
	feat=[]
	#feat_int=[0]
	A_meta = np.zeros((DF_meta.shape[0],0))
	for c in DF_meta.columns:
		if c!='elevation':
			category = list(set(DF_meta[c]))
			AA= np.zeros((DF_meta.shape[0],len(category)))
			for i,ca in enumerate(category):
				AA[DF_meta[c]==ca,i]=1
			A_meta = np.hstack((A_meta,AA))
			#feat_int.append(len(category))
			for ca in category:
				feat.append(c+'|'+str(ca))
	# ELEVATION has Not provided values, how to replace?
	#feat.append('elevation')
	#A_meta = np.hstack((A_meta,DF_meta['elevation'].values()))
	#feat_int = np.cumsum(np.array(feat_int))
	return feat,A_meta

def xgb_select_feat(A,feat,y,t,DF_meta):
	label  = np.array(y)
	quest = t.split('|')[0]
	filt_noresp = DF_meta[quest]!='Not provided'
	AA = A[filt_noresp]
	labell = label[filt_noresp]
	sec = []
	start_time = time.time()
	model = XGBClassifier(max_depth=2,learning_rate=0.2,n_jobs=7,min_child_weight=5,reg_alpha=10,reg_lambda=20,n_estimators=20)
	model.fit(AA, labell)
	print('Fit time: '+str(time.time()-start_time))
	res = model.feature_importances_
	sec = list(res.argsort()[-5:][::-1])
	return sec

def xgb_run_with_sel_feats(A,sec,taxacols,feat,y,t,fo,DF_meta,shan):#
	label  = np.array(y)
	quest = t.split('|')[0]
	filt_noresp = DF_meta[quest]!='Not provided'
	AA = A[filt_noresp]
	labell = label[filt_noresp]
	shann = shan[filt_noresp]
	dtrain = xgb.DMatrix(AA, label=labell)
	param = {'max_depth': 2,
		  'min_child_weight': 5,
		  'subsample': 1,
		  'eta': 0.2,
		  'alpha': 10,
		  'lambda': 20,
		  'silent': 1,
		  'objective': 'binary:logistic',
		  'nthread':7,
		  'eval_metric':'auc'}
	num_round = 300

	for _ in range(100):
		try:
			start_time = time.time()
			rs = np.random.randint(100)
			res = xgb.cv(param, dtrain, num_round, nfold=5,metrics={'auc'}, seed=rs,verbose_eval=False,show_stdv=False)
			break
		except:
			pass
#			start_time = time.time()
#			rs = np.random.randint(100)
#			res = xgb.cv(param, dtrain, num_round, nfold=5,metrics={'auc'}, seed=rs,verbose_eval=False,show_stdv=False)
	qq=res.loc[res['test-auc-mean'].argmax()]
	fo.write(t+'=======================:\n')
	pva,meanst = kw_alpha(shann,labell)
	pvalg,meanstlg = kw_alpha(np.log2(shann),labell)
	fo.write('Alpha_div:\n')
	fo.write('PVAL\t'+pva+'\n')
	fo.write('STAT TRUE\t'+meanst[1]+'\n')
	fo.write('STAT FALSE\t'+meanst[0]+'\n')
#	fo.write('PVAL\t'+pva+'\t'+pvalg+'\n')
#	fo.write('STAT TRUE\t'+meanst[1]+'\t'+meanstlg[1]+'\n')
#	fo.write('STAT FALSE\t'+meanst[0]+'\t'+meanstlg[0]+'\n')
	fo.write('Selected OTU:\t')
	fo.write('\t'.join([taxacols[i] for i in sec])+'\n')
	try:
		pval,meanstd = t_test_stat(AA,sec,labell)
		fo.write('PVAL\t'+'\t'.join(pval)+'\n')
		fo.write('STAT TRUE\t'+'\t'.join([ms[1] for ms in meanstd])+'\n')
		fo.write('STAT FALSE\t'+'\t'.join([ms[0] for ms in meanstd])+'\n')
	except:
		pass
	fo.write('test-auc\ttest-auc-std\ttrain-auc\ttrain-auc-std\n')
	fo.write(str(qq['test-auc-mean'])+'\t'+str(qq['test-auc-std'])+'\t'+str(qq['train-auc-mean'])+'\t'+str(qq['train-auc-std'])+'\n')
	
	dtrain = xgb.DMatrix(AA[:,sec[0]:sec[0]+1], label=labell)
	res = xgb.cv(param, dtrain, num_round, nfold=5,metrics={'auc'}, seed=rs,verbose_eval=False,show_stdv=False)
	qq=res.loc[res['test-auc-mean'].argmax()]
	fo.write('test-auc\ttest-auc-std\ttrain-auc\ttrain-auc-std\n')
	fo.write(str(qq['test-auc-mean'])+'\t'+str(qq['test-auc-std'])+'\t'+str(qq['train-auc-mean'])+'\t'+str(qq['train-auc-std'])+'\n')
	
	
	dtrain = xgb.DMatrix(AA[:,sec], label=labell)
	res = xgb.cv(param, dtrain, num_round, nfold=5,metrics={'auc'}, seed=rs,verbose_eval=False,show_stdv=False)
	qq=res.loc[res['test-auc-mean'].argmax()]
	fo.write('test-auc\ttest-auc-std\ttrain-auc\ttrain-auc-std\n')
	fo.write(str(qq['test-auc-mean'])+'\t'+str(qq['test-auc-std'])+'\t'+str(qq['train-auc-mean'])+'\t'+str(qq['train-auc-std'])+'\n')
	
	nofeat = [f for f in range(AA.shape[1]) if f not in sec]
	dtrain = xgb.DMatrix(AA[:,nofeat], label=labell)
	res = xgb.cv(param, dtrain, num_round, nfold=5,metrics={'auc'}, seed=rs,verbose_eval=False,show_stdv=False)
	qq=res.loc[res['test-auc-mean'].argmax()]
	fo.write('ttest-auc\ttest-auc-std\ttrain-auc\ttrain-auc-std\n')
	fo.write(str(qq['test-auc-mean'])+'\t'+str(qq['test-auc-std'])+'\t'+str(qq['train-auc-mean'])+'\t'+str(qq['train-auc-std'])+'\n\n')
	print('CV time: '+str(time.time()-start_time))
	return res

def t_test_stat(A,sec,label):
	pval=[]
	meanstd=[]
	otuno = A.shape[1]
	for s in sec:
		x = A[label==0,s]
		y = A[label==1,s]
		meanstd.append([str(x.mean())+'+-'+str(x.std()),str(y.mean())+'+-'+str(y.std())])
		pval.append(str(stats.kruskal(x,y)[1]*otuno))
	return pval,meanstd

def alpha_div(A):
	AA=A.copy()
	AA[A==0]=1
	return -(A*np.log2(AA)).sum(axis=1)

def kw_alpha(shan,label):
	x = shan[label==0]
	y = shan[label==1]
	meanstd = (str(x.mean())+'+-'+str(x.std()),str(y.mean())+'+-'+str(y.std()))
	pval = str(stats.kruskal(x,y)[1])
	return pval,meanstd

def prepare_disease(dis,dislist,DF,cat_feats):
	DFF = DF.drop([d for d in dislist if d!= dis],axis=1)
	if 'allergi' not in dis and dis!='chickenpox' and dis!='lactose' and dis!='tonsils_removed':
		crit1 = DFF[dis] =='Diagnosed by a medical professional (doctor, physician assistant)'
		crit2 = DFF[dis]=='I do not have this condition'
	else:
		crit1 = DFF[dis] =='T'
		crit2 = DFF[dis] =='F'
	idx = crit1|crit2
	return DFF.loc[idx],[c for c in cat_feats if c not in dislist and c!=dis]

def prepare_disease_eq(dis,dislist,DF,cat_feats,mult):
	DFF = DF.drop([d for d in dislist if d!= dis],axis=1)
	if 'allergi' not in dis and dis!='chickenpox' and dis!='lactose' and dis!='tonsils_removed':
		crit1 = DFF[dis] =='Diagnosed by a medical professional (doctor, physician assistant)'
		crit2 = DFF[dis]=='I do not have this condition'
	else:
		crit1 = DFF[dis] =='T'
		crit2 = DFF[dis] =='F'
	idx1 = list(DFF.loc[crit1].index)
	idx2 = DFF.loc[crit2].index
	sec=np.random.permutation(len(idx2))[:mult*len(idx1)]
	idx2=list(idx2[sec])
	return DFF.loc[idx1+idx2],[c for c in cat_feats if c not in dislist and c!=dis]

def find_cat(feats):
	return [f for f in feats if (f!='bmi_corrected' and f!='age_years' and f!='elevation')]

def cat_embsize(DF_meta,feats):
	cat_dict={}
	for f in feats:
		cat_dict[f] = min((len(set(DF_meta[f]))+1)//2,50)
	return cat_dict

def testtrain(df,dis):
	C = list(set(df[dis]))
	if len(C)>1:
		idx1 = df.loc[df[dis]==C[0]].index
		idx2 = df.loc[df[dis]==C[1]].index
		l1 = len(idx1)
		l2 = len(idx2)
		return list(idx1[np.random.permutation(l1)[:l1//10]])+list(idx2[np.random.permutation(l2)[:l2//10]])
	else:
		return -1

def testtrain_eq(df,dis):
	C = list(set(df[dis]))
	if len(C)>1:
		idx1 = df.loc[df[dis]==C[0]].index
		idx2 = df.loc[df[dis]==C[1]].index
		l1 = len(idx1)
		l2 = len(idx2)
		l = min(l1,l2)
		return list(idx1[np.random.permutation(l)[:l//5]])+list(idx2[np.random.permutation(l)[:l//5]])
	else:
		return -1

def testtrain_eq_age_corrected(df,dis):
	C = list(set(df[dis]))
	if len(C)>1:
		idx2 = df.loc[df[dis]==C[1]].index
		idx1 = df.loc[df[dis]==C[0]].index
		if len(idx2)>len(idx1):
			sw=idx1.copy()
			idx1=idx2.copy()
			idx2=sw.copy()
		idx1 = list(idx1[np.random.permutation(len(idx1))])
		l2 = len(idx2)
		idx2 = list(idx2[np.random.permutation(l2)[:l2//5]])
		idx=[]
		active = set()
		yaslar = df.loc[idx1]['age_cat']
		for i in idx2:
			age_cat = df.loc[i]['age_cat']
			for a in idx1:
				if yaslar[a]==age_cat and a not in active:
					idx.append(a)
					active.add(a)
					break
		return idx+idx2
	else:
		return -1



def fastai_acc(learn,data,tsize,lrn):
	learn.fit(1, lrn)
	acc = learn.validate()[1]
	a = learn.get_preds()
	t = np.array(a[1])
	spn = (t==0).sum()
	snn = len(t)-spn
	p = np.array(a[0][:,1])
	auc,roc = auc_calc(p,t,spn,snn)
	sn=0
	sp=0
	for _ in range(10):
		learn.fit(1, 1e-2)
		acc_new = learn.validate()[1]
		a = learn.get_preds()
		t = np.array(a[1])
		spn = (t==0).sum()
		snn = len(t)-spn
		p = np.array(a[0][:,1])
		sn_new = t[p>=0.5].sum()/snn
		sp_new = (t[p<0.5]==0).sum()/spn
		auc_new,roca = auc_calc(p,t,spn,snn)
		sc_val=np.array(learn.get_preds(ds_type=data.train_ds)[0][:,1])
		if auc_new > auc:
			auc=auc_new
			acc=acc_new
			roc = roca
			sn = sn_new
			sp=sp_new
	return auc,acc,sc_val,sn,sp

def auc_calc(p,t,spn,snn):
	idx = np.argsort(p)
	p=p[idx]
	t=t[idx]
	lena = spn+snn
	roc = np.zeros((lena,2))
	for i in range(lena):
		sp = (t[p<p[i]]==0).sum()/spn
		sn = 1-t[p>=p[i]].sum()/snn
		roc[i,0]=sp
		roc[i,1]=sn
	roc[-1]=[1.0,1.0]
	dif1 = roc[1:,0]-roc[:-1,0]
	dif2 = roc[1:,1]+roc[:-1,1]
	return 1-(dif1*dif2).sum()/2,roc

def disease_transfer(dis_vec,df,roc,valid_idx):#df_dis.drop(df_dis.loc[valid_idx].index)
	df=df.drop(df.loc[valid_idx].index)
	df = df.reset_index()
	df.drop('index',axis=1,inplace=True)
	accur=[]
	aucc =[]
	current_diseases = list(df.columns)
	param = {'max_depth': 2,
		  'min_child_weight': 2,
		  'subsample': 1,
		  'eta': 0.2,
		  'silent': 1,
		  'objective': 'binary:logistic',
		  'nthread':7,
		  'eval_metric':'auc'}
	for i,d in enumerate(dis_vec):
		if d in current_diseases:
			t=-np.ones(df.shape[0])
			if 'allergi' not in d and d!='chickenpox' and d!='lactose' and d!='tonsils_removed':
				idx1 = list(df.loc[df[d]=='Diagnosed by a medical professional (doctor, physician assistant)'].index)
				idx2 = list(df.loc[df[d]=='I do not have this condition'].index)
			else:
				idx1 = list(df.loc[df[d]=='T'].index)
				idx2 = list(df.loc[df[d]=='F'].index)
			t[idx1]=1
			t[idx2]=0
			idx=idx1+idx2
			t=t[idx]
			spn = (t==0).sum()
			snn = len(t)-spn
			print(spn,snn)
			if spn>10 and snn>10:
				dtrain = xgb.DMatrix(roc[idx].reshape((len(t),1)), label=t)
				res = xgb.cv(param,dtrain, 10, nfold=2,metrics={'error','auc'}, seed=42,verbose_eval=False,show_stdv=False)
				accur.append(1-res['test-error-mean'].max())
				q,qq = auc_calc(roc[idx],t,spn,snn)
				aucc.append(1-q)
			else:
				aucc.append(0)
				accur.append(0)
		else:
			aucc.append(-1)
			accur.append(-1)
	return aucc,accur

def disease_transfer_eq(dis_vec,df,roc,valid_idx):#df_dis.drop(df_dis.loc[valid_idx].index)
	df=df.drop(df.loc[valid_idx].index)
	df = df.reset_index()
	df.drop('index',axis=1,inplace=True)
	accur=[]
	aucc =[]
	current_diseases = list(df.columns)
	param = {'max_depth': 2,
		  'min_child_weight': 2,
		  'subsample': 1,
		  'eta': 0.2,
		  'silent': 1,
		  'objective': 'binary:logistic',
		  'nthread':7,
		  'eval_metric':'auc'}
	for i,d in enumerate(dis_vec):
		if d in current_diseases:
			t=-np.ones(df.shape[0])
			if 'allergi' not in d and d!='chickenpox' and d!='lactose' and d!='tonsils_removed':
				idx1 = list(df.loc[df[d]=='Diagnosed by a medical professional (doctor, physician assistant)'].index)
				idx2 = list(df.loc[df[d]=='I do not have this condition'].index)
			else:
				idx1 = list(df.loc[df[d]=='T'].index)
				idx2 = list(df.loc[df[d]=='F'].index)
			l1,l2=len(idx1),len(idx2)
			if l1>l2:
				secc = np.random.permutation(l1)
				idx1 = [idx1[j] for j in secc[:l2]]
			else:
				secc = np.random.permutation(l2)
				idx2 = [idx2[j] for j in secc[:l1]]
			t[idx1]=1
			t[idx2]=0
			idx=idx1+idx2
			t=t[idx]
			spn = (t==0).sum()
			snn = len(t)-spn
			print(spn,snn)
			if spn>10 and snn>10:
				dtrain = xgb.DMatrix(roc[idx].reshape((len(t),1)), label=t)
				res = xgb.cv(param,dtrain, 10, nfold=2,metrics={'error','auc'}, seed=42,verbose_eval=False,show_stdv=False)
				accur.append(1-res['test-error-mean'].max())
				q,qq = auc_calc(roc[idx],t,spn,snn)
				aucc.append(1-q)
			else:
				aucc.append(0)
				accur.append(0)
		else:
			aucc.append(-1)
			accur.append(-1)
	return aucc,accur

def drop_cat(dis,df,cat_var):
	cat_var = [c for c in cat_var if c!='age_cat' and c!='sex' and c!=dis]
	df = df.drop(cat_var,axis=1)
	return df,['age_cat','sex']

