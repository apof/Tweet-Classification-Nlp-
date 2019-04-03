import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import math
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import classification_report


def load_dataset(dir_name):
    df = pd.read_csv(dir_name, sep="\t")
    df.columns = ['a', 'b','label','tweet']
    #print("Shape of pandas frame: " + str(df.shape))
    return df['tweet'],df['label']

def split_into_train_test(vectors,labels):
	train_num = int(math.floor(0.9*len(vectors)))
	return vectors[0:train_num],labels[0:train_num],vectors[train_num:len(vectors)],labels[train_num:len(vectors)]

def KfoldCrossValidation(clf,inputs, labels, folds):

	kf = KFold(n_splits=folds)

	accuracy = 0
	precision = 0
	recall = 0
	f1 = 0

	count = 1
	
	for train_index, test_index in kf.split(inputs):
		X_train, X_test = inputs[train_index], inputs[test_index]
		y_train, y_test = labels[train_index], labels[test_index]

		clf.fit(X_train,y_train)
		preds = clf.predict(X_test)
		print("-->Fold: " + str(count) + " in progress..")

		accuracy += accuracy_score(y_test, preds)
		precision += precision_score(y_test, preds,average='micro')
		recall += recall_score(y_test, preds,average='micro')
		f1 += f1_score(y_test, preds,average='micro')


		count += 1

	print("Total Metrics after all folds are: ")
	print("Accuracy = " + str(accuracy/folds))
	print("Recall = " + str(recall/folds))
	print("Precision = "  + str(precision/folds))
	print("F1 = " + str(f1/folds))

def calculate_metrics(y_pred,y_true):
	target_names = ['Negative', 'Neutral', 'Positive']
	print(classification_report(y_true, y_pred, target_names=target_names))
	print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))

def dataset_balance(labels):

	positive = 0
	negative = 0
	neutral = 0

	for l in labels:
		if (l==0):
			negative += 1
		elif(l==1):
			neutral += 1
		else:
			positive += 1

	print('positive: ' + str(positive))
	print('negative: ' + str(negative))
	print('neutral: ' + str(neutral))

