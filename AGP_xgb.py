import utils
import xgboost as xgb
import sys
import pandas as pd
import numpy as np


# INPUT
metadata = 'metadata_gut_nr_noft.csv' # sys.argv[1]
otufile = 'OTU_TABLE_gut_nozero.csv' # sys.argv[2]

clade_level = 's' # sys.argv[3]
bloom = True # sys.argv[4]
resample = False
thres = 1000
count = 100000
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

#filter metadata with no sequences
DF_meta = utils.filter_nonsequenced_samples(idrows,DF_meta)

# =========================================== REDUCE METADATA ======
#remove samples with wrong bmi
idx=list(range(DF_meta.shape[0]))
#idx,bmi,DF_meta = utils.filter_bmi(DF_meta)

#pick selected features
DF_meta = utils.wanted_metadata(DF_meta)

#remove features with  unbalances class distribution
DF_meta = utils.remove_unbalanced_features(DF_meta)

#filter nonexisting samples from sequence data
A, idrows = utils.remove_otus_unwanted_metadata(idx,A,idrows)

# onehot encoding of metadata
feat,A_meta = utils.metadata_onehot(DF_meta)
shan = utils.alpha_div(A)
print(A_meta.shape[1])
fo = open('aucresults/auc_genus.csv','w')
for i in range(A_meta.shape[1]):#range(5):#
	print(i)
	t = feat[i]
	if not 'Not provided' in t:
		sec = utils.xgb_select_feat(A,feat,A_meta[:,i],t,DF_meta)
		res = utils.xgb_run_with_sel_feats(A,sec,taxacols,feat,A_meta[:,i],t,fo,DF_meta,shan)
fo.close()

#A = np.hstack((A,A_meta))
#feat = taxacols+feat














