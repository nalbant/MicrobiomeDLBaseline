import tensorflow as tf
import keras
import numpy as np
from keras import models
from keras import layers
from keras import losses, optimizers, metrics, activations
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Conv1D, MaxPooling1D,  GlobalMaxPooling1D
import sys

import sklearn.metrics as M

fnam = sys.argv[1]
otus = np.genfromtxt(fnam, delimiter=',')

final_otus = np.transpose(otus)

feat_data = final_otus[:,1:]
labels = final_otus[:,0]

acc=[]
roc=[]
f1=[]

for i in range(10):
	rs = np.random.randint(100)
	X_train, X_test, y_train, y_test = train_test_split(feat_data, labels,
	                                                    test_size=0.2,
	                                                    random_state=rs)



	# scaler = MinMaxScaler()
	# scaled_x_train = scaler.fit_transform(X_train)
	scaled_x_train = X_train 
	scaled_x_test = X_test
	indim = X_train.shape[1]

	# scaled_x_test = scaler.transform(X_test)



	dnn_keras_model = models.Sequential()

	#dnn_keras_model.add(layers.Dropout(0.5, input_dim=123))

	nneu= 256

	dnn_keras_model.add(layers.Dense(units=nneu, input_dim=indim, activation='selu'))
	#dnn_keras_model.add(layers.Dense(units=1, input_dim=indim, activation='sigmoid'))
	for i in range(4):
		dnn_keras_model.add(layers.Dense(units=nneu, activation='selu'))
		#dnn_keras_model.add(BatchNormalization())
		#dnn_keras_model.add(layers.Dropout(0.5))
		nneu//=2
		#dnn_keras_model.add(layers.Dropout(0.6))
	# dnn_keras_model.add(layers.Dense(units=256, activation='selu'))
	# #dnn_keras_model.add(layers.Dropout(0.5))
	# dnn_keras_model.add(layers.Dense(units=10, activation='selu'))
	# #dnn_keras_model.add(layers.Dropout(0.5))
	# 
	dnn_keras_model.add(layers.Dense(units=1, activation='sigmoid'))
	dnn_keras_model.compile(optimizer='adam',
	                       loss='binary_crossentropy',
	                       metrics=['accuracy'])

	dnn_keras_model.fit(scaled_x_train,y_train, verbose=0, epochs=50)   #batch_size=1, 

	preds = dnn_keras_model.predict_classes(scaled_x_test)
	print(M.accuracy_score(preds, y_test))
	acc.append(M.accuracy_score(preds, y_test))
	roc.append(M.roc_auc_score(preds, y_test))
	f1.append(M.f1_score(preds, y_test))


print('Mean acc: '+str(np.array(acc).mean())+'\n')
print('Mean roc: '+str(np.array(roc).mean())+'\n')
print('Mean f1: '+str(np.array(f1).mean())+'\n')


#print(classification_report(preds, y_test))


