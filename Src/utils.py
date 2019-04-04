import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import math
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids


def load_dataset(dir_name):
    df = pd.read_csv(dir_name, sep="\t")
    df.columns = ['a', 'b','label','tweet']
    #print("Shape of pandas frame: " + str(df.shape))
    return df['tweet'],df['label']

def split_into_train_test(vectors,labels):
	#train_num = int(math.floor(0.7*len(vectors)))
	#return vectors[0:train_num],labels[0:train_num],vectors[train_num:len(vectors)],labels[train_num:len(vectors)]
	train_data, test_data, train_labels, test_labels = train_test_split(vectors, labels, test_size=0.3, random_state=42)
	return train_data,train_labels,test_data,test_labels

def KfoldCrossValidation(clf,inputs, labels, folds):

	kf = KFold(n_splits=folds)

	accuracy = 0
	precision = 0
	recall = 0
	f1 = 0

	count = 0
	
	for train_index, test_index in kf.split(inputs):
		X_train, X_test = inputs[train_index], inputs[test_index]
		y_train, y_test = labels[train_index], labels[test_index]

		clf.fit(X_train,y_train)
		preds = clf.predict(X_test)
		print("-->Fold: " + str(count) + " in progress..")

		#accuracy += accuracy_score(y_test, preds)
		#precision += precision_score(y_test, preds,average='micro')
		#recall += recall_score(y_test, preds,average='micro')
		current_f1 = f1_score(y_test, preds,average='micro')
		f1 += current_f1

		count += 1

		#print("Accuracy = " + str(accuracy/count))
		#print("Recall = " + str(recall/count))
		#print("Precision = "  + str(precision/count))
		print("total F1 = " + str(f1/count))
		print("current F1 = " + str(current_f1))

def calculate_metrics(y_pred,y_true):
	target_names = ['Negative', 'Neutral', 'Positive']
	print(classification_report(y_true, y_pred, target_names=target_names))
	print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# balance the dataset making over-samlping (replication) and subsampling
def dataset_balance(labels,tweets):

	positive_list = []
	negative_list = []
	neutral_list = []

	positive = 0
	negative = 0
	neutral = 0

	count = 0
	for l in labels:
		if (l==0):
			negative += 1
			negative_list.append(tweets[count])
		elif(l==1):
			neutral += 1
			neutral_list.append(tweets[count])
		else:
			positive += 1
			positive_list.append(tweets[count])
		count += 1

	#print('positive: ' + str(positive) + ' ' + str(len(positive_list)))
	#print('negative: ' + str(negative) + ' ' + str(len(negative_list)))
	#print('neutral: ' + str(neutral) + ' ' + str(len(neutral_list)))

	n = 2000
	negative_data = np.array(negative_list)
	random_negative = negative_data[np.random.choice(len(negative_data), size=n, replace=False)]

	negative_data = np.concatenate((negative_data,random_negative),axis = 0)

	n = 6465
	positive_data = np.array(positive_list)
	neutral_data = np.array(neutral_list)
	random_positive = positive_data[np.random.choice(len(positive_data), size=n, replace=False)]
	random_neutral = neutral_data[np.random.choice(len(neutral_data), size=n, replace=False)]

	neutral_data = random_neutral
	positive_data = random_positive

	np.random.shuffle(neutral_data)
	np.random.shuffle(positive_data)
	np.random.shuffle(negative_data)

	negative_labels = np.zeros(n)
	neutral_labels = np.ones(n)
	positive_labels =  np.full(n,2)

	labels = np.concatenate((negative_labels,neutral_labels),axis = 0)
	labels = np.concatenate((labels,positive_labels),axis = 0)

	vectors = np.concatenate((negative_data,neutral_data),axis = 0)
	vectors = np.concatenate((vectors,positive_data),axis = 0)

	return unison_shuffled_copies(vectors,labels)


def dataset_sampling(X,y):
	sm = SMOTE(random_state=42,ratio='minority')
	smt = SMOTETomek(ratio='auto')
	ros = RandomOverSampler(random_state=0)
	rus = RandomUnderSampler(random_state=0)
	tl = TomekLinks(return_indices=True, ratio='majority')
	cc = ClusterCentroids(ratio={0: 10})
	#X_res, y_res = sm.fit_resample(X, y)
	#X_res, y_res = ros.fit_resample(X, y)
	#X_res, y_res = rus.fit_resample(X, y)
	X_res, y_res, id_tl = tl.fit_sample(X, y)
	#X_res, y_res = cc.fit_sample(X, y)
	#X_res, y_res = smt.fit_sample(X, y)
	return X_res,y_res