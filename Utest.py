import scipy.stats as st
import numpy as np
import utils
import pandas as pd


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

bacter = ['anaerostipes','akkermansia','bifidobacterium','coprococcus','eubacterium','roseburia','ruminococcus','streptococcus','bacillus','lactobacillus','faecalibacterium','holdemania','subdomigranilum','anaerofilum','oscillibacter','dorea','blautia','megasphera','bacteroides','streptococcus','butyrivibrio','prevotella']

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

DF = pd.DataFrame(A,columns = taxacols, index = idrows)

# =========================================== REDUCE METADATA ======

DF_meta = utils.remove_unbalanced_features(DF_meta)
df =  pd.concat([DF_meta, DF], axis=1)
df = df.reset_index()

fets=list(df.columns)
bbb=fets[226:]
bb=[b.split('g__')[-1].lower() for b in bbb]
bacterlist=[]
for bac in bacter:
	if bac in bb:
		bacterlist.append((bac,bb.index(bac)+226))
# ==============================
th=0.01


basl = fets[197]
print(basl)
c=list(set(df[basl]))
print(c)

c0= c[7]
c1= c[40]

fo=open('../ubiome2diet/'+basl+'.csv','w')
fo.write('tax\tp\tu\tavg_'+c0+'\tavg_'+c1+'\n')
idx0 = df[basl]==c0
idx1 = df[basl]==c1
for b in bacterlist:
	x=np.array(df[fets[b[1]]].loc[idx0])
	y=np.array(df[fets[b[1]]].loc[idx1])
	u=np.inf
	p=np.inf
	try:
		u,p = st.mannwhitneyu(x,y)
		u=1-u/(len(x)*len(y))
		p*=1299
	except:
		pass
	if p<th:
		fo.write(b[0]+'\t'+str(p)+'\t'+str(u)+'\t'+str(x.mean())+'\t'+str(y.mean())+'\n')
fo.write('\nSELECTED\n')
lili=np.zeros((0,4))
secf=[]
for i in range(226,len(fets)):
	x=np.array(df[fets[i]].loc[idx0])
	y=np.array(df[fets[i]].loc[idx1])
	u=np.inf
	p=np.inf
	try:
		u,p = st.mannwhitneyu(x,y)
		u=1-u/(len(x)*len(y))
		p*=1299
	except:
		pass
	if p<th:
		secf.append(fets[i])
		lili=np.vstack((lili,np.array([p,u,x.mean(),y.mean()])))
idx = np.argsort(lili[:,0])
for i in idx:
	fo.write(secf[i]+'\t'+str(lili[i,0])+'\t'+str(lili[i,1])+'\t'+str(lili[i,2])+'\t'+str(lili[i,3])+'\n')
fo.close()

clear




























