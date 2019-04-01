import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import math
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix

def load_dataset(dir_name):
    df = pd.read_csv(dir_name, sep="\t")
    df.columns = ['a', 'b','label','tweet']
    #print("Shape of pandas frame: " + str(df.shape))
    return df['tweet'],df['label']

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

