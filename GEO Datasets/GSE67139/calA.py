# Simple Python script to classify using a reduced number of features
# by Alberto Tonda, 2018 <alberto.tonda@gmail.com>

import copy
import numpy as np
import os
import sys
import math 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
import pandas as pd 
from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import StandardScaler

# used for normalization
from sklearn.preprocessing import MaxAbsScaler
# used for normalization
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import QuantileTransformer
# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# used to compute accuracy
from sklearn.metrics import accuracy_score

# this is an incredibly useful function
from pandas import read_csv


def loadTrainingAndTest() :
	
	name='GSE67139'
	
	
	# data used for the predictions
	dfData = read_csv("./TCGA/data.csv", header=None, sep=',')
	dfLabels = read_csv("./TCGA/labels.csv", header=None)
	dfFeatures = read_csv("./TCGA/features.csv", header=None)		
	df_training = dfData.values
	trainingFeatures = dfFeatures.values[:,0]
	y_training = dfLabels.values[:,0]
	# load test as dataframe
	dfDataTest = read_csv("./data/dataTest.csv", header=None, sep=',')
	dfLabelsTest = read_csv("./data/labelsTest.csv", header=None)
	dfFeaturesTest = read_csv("./data/featuresTest.csv", header=None)		
	df_test = dfDataTest.values
	testFeatures = dfFeaturesTest.values[:,0]
	y_test = dfLabelsTest.values[:,0]
	
	
	
	print(len(df_training))
	print(len(df_training[0]))
	print(len(trainingFeatures))
	print(len(y_training))
	
	print(len(df_test))
	print(len(df_test[0]))
	print(len(testFeatures))
	print(len(y_test))

	# load reduced set of features
	commonFeatures = dfFeatures.values[:,0]
	
	commonFeatures = [ f for f in commonFeatures if f in trainingFeatures and f in testFeatures ]
	print("Training set has %d features; test set has %d features. The two datasets have a total of %d common features (among the top 100 most relevant)." % (len(trainingFeatures), len(testFeatures), len(commonFeatures)))

	#print(commonFeatures)

	X_training=np.zeros(shape=(len(y_training), len(commonFeatures)))
	X_test=np.zeros(shape=(len(y_test), len(commonFeatures)))
	
	for i in range(0,len(commonFeatures)):
		for j in range (0,len(trainingFeatures)):
			if (commonFeatures[i]==trainingFeatures[j]):
				for k in range (0,len(y_training)):
					X_training[k,i]=df_training[k,j]
	
	for i in range(0,len(commonFeatures)):
		for j in range (0,len(testFeatures)):
			if (commonFeatures[i]==testFeatures[j]):
				for k in range (0,len(y_test)):
					X_test[k,i]=df_test[k,j]
		
	print(len(X_training))
	print(len(X_training[0]))
	
	print(len(X_test))
	print(len(X_test[0]))
	
	#pd.DataFrame(X_training).to_csv("./data/X_training.csv", header=None, index =None)
	#pd.DataFrame(X_test).to_csv("./data/X_test.csv", header=None, index =None)
	
	# instead, compute centroid of training set and add it (feature-wise) to test set
	std_training = np.zeros( X_training.shape[1] )
	centroid_training = np.zeros( X_training.shape[1] )
	for j in range(0, X_training.shape[1]) :
		centroid_training[j] = np.average( X_training[:,j] )
		std_training[j] = np.std( X_training[:,j] )

	# instead, compute centroid of training set and add it (feature-wise) to test set
	std_test = np.zeros( X_test.shape[1] )
	centroid_test = np.zeros( X_test.shape[1] )
	for j in range(0, X_test.shape[1]) :
		centroid_test[j] = np.average( X_test[:,j] )
		std_test[j] = np.std( X_test[:,j] )
	#

	#X_test=np.power(X_test,2)
	#scaler= MaxAbsScaler()
	#X_test=scaler.fit_transform(X_test)
	
	#scaler=MinMaxScaler()
	#scaler.fit(X_training)
	#X_test=scaler.inverse_transform(X_test)

	#value=46227
	value=203580
	#50 chips
	X_test = ((X_test)*(std_training/np.sqrt(1)))+centroid_training
	#X_test=X_test*centroid_training


	
	classValue=(y_test[0])
	
	return X_training, X_test, y_training, y_test, commonFeatures, classValue, name
def saveMatrix(name,var):
	np.savetxt(name, var, fmt='%1.3f', delimiter=',')
	
def main() :
	
	X_training, X_test, y_training, y_test, featureNames, classValue, name = loadTrainingAndTest()
	
	print("Pre-processing data...")
	
	dfFeaturesTest = read_csv("./data/coefs.csv", header=None)
	
	values = dfFeaturesTest.values[:,0]
	
	print(values)	
	for i in range (0,len(X_test[0])):
		for j in range(0,len(X_test)):
			X_test[j,i]=X_test[j,i]*values[i]
	
	if True :
	# let's normalize the data by sample
		scaler_sample = StandardScaler()
		scaler_sample2 = StandardScaler()
		X_training = scaler_sample.fit_transform(X_training.T).T
		X_test = scaler_sample2.fit_transform(X_test.T).T 
	
	if True :
		scaler = StandardScaler()
		X_training = scaler.fit_transform(X_training)
		X_test = scaler.transform(X_test)
	
	
	
	
	# also normalize by feature
	
	# time to plot a PCA
	if True :
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		pca.fit(X_training)
		X_pca_train = pca.transform(X_training)
		X_pca_test = pca.transform(X_test)
		
		import matplotlib.pyplot as plt
		fig = plt.figure()
		ax = fig.add_subplot(111)
		
		class3 = [ (y == classValue) for y in y_training ]
		
		
		ax.plot(X_pca_train[:,0], X_pca_train[:,1], 'b.', label="tcga data")
		ax.plot(X_pca_train[class3,0], X_pca_train[class3,1], 
		color='orange', marker='.', linestyle='None', label=("tcga data, class "+str(classValue)))
		ax.plot(X_pca_test[:,0], X_pca_test[:,1], 'r.', label=name+" data")
		
		ax.legend(loc='best')
		ax.set_title("TCGA vs "+ name +" data")
		ax.set_xlabel("PCA 0")
		ax.set_ylabel("PCA 1")
		
		plt.savefig(name+".png")
	
	results=np.zeros(shape=(X_test.shape[0]*9,10))
	results2=np.zeros(shape=(9,10))
	j=0
	for k in range (0,10):
		
		## FINALLY, WE CAN CLASSIFY AWAY!
		classifierList = [

				#[RandomForestClassifier(), "RandomForestClassifier()"],
				
				
				[LogisticRegression(), "LogisticRegression"], # coef_
				[PassiveAggressiveClassifier(), "PassiveAggressiveClassifier"], # coef_
				[SGDClassifier(), "SGDClassifier"], # coef_
				[SVC(kernel='linear'), "SVC(linear)"], # coef_, but only if the kernel is linear...the default is 'rbf', which is NOT linear
				[RidgeClassifier(), "RidgeClassifier"], # coef_
				[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],
				[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
				[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
				]
		

				
		f2 = open(name+"_"+str(k)+"_.txt", 'w')
		f = open(name+"_"+str(k)+".txt", 'w')
		l=0;
		##for i in range(0, 10) :
		for originalClassifier, classifierName in classifierList :
			f.write("\nClassifier " + classifierName+ "\n")
			f2.write("\nClassifier " + classifierName+ "\n")
			print("\nClassifier " + classifierName)
			##let's normalize, anyway
			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_training, y_training)
			y_train_pred = classifier.predict(X_training)
			y_test_pred = classifier.predict(X_test)

			scoreTraining = accuracy_score(y_train_pred, y_training)
			scoreTest = accuracy_score(y_test_pred, y_test)
			
			f.write("Training accuracy: %.4f; Test accuracy: %.4f \n" % (scoreTraining, scoreTest))
			f2.write("Training accuracy: %.4f; Test accuracy: %.4f \n" % (scoreTraining, scoreTest))
			print("Training accuracy: %.4f; Test accuracy: %.4f" % (scoreTraining, scoreTest))
			f.write("Complete classification on test: \n")
			for i in range(0, y_test.shape[0]) :
				f.write("%d \n" % (y_test_pred[i]))
				results[j,k]=y_test_pred[i]
				j=j+1
			results2[l,k]=scoreTest
			l=l+1
		f.close()
		f2.close()
		j=0
	saveMatrix(name+"_results.csv",results)
	saveMatrix(name+"results2.csv",results2)
	return


if __name__ == "__main__" :
	sys.exit( main() )
