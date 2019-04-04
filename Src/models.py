import preprocessing
import utils
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


sentiment_dicts =  preprocessing.convert_lex_to_dict()

tweets,labels = utils.load_dataset('../twitter_data/train2017.tsv')

encoded_labels = preprocessing.one_hot_encode(labels)

tweets = preprocessing.exclude_words(tweets)

list_of_processed_tweets = preprocessing.process_tweets(tweets)

sentiment_vectors = preprocessing.create_sentiment_vectors(list_of_processed_tweets,sentiment_dicts)
vectorized_tweets = preprocessing.tfidf_vectorization(list_of_processed_tweets)
dim_reduced_tweets = preprocessing.dim_reduction(vectorized_tweets)
final_vectors = preprocessing.concatenate_vectors(dim_reduced_tweets,sentiment_vectors)

final_vectors,encoded_labels =  utils.dataset_balance(encoded_labels,final_vectors)

#clf = KNeighborsClassifier(n_neighbors=11,metric = 'cosine')
#clf = SVC(gamma='auto',kernel='poly')
#clf = SGDClassifier(max_iter=1000, tol=1e-3)
#clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,random_state=1)
#train_data,train_labels,test_data,test_labels = utils.split_into_train_test(final_vectors,encoded_labels)
#clf.fit(train_data,train_labels)
#preds = clf.predict(test_data)
#utils.calculate_metrics(test_labels,preds)
print("10FoldCross Validation...")
utils.KfoldCrossValidation(clf,np.array(final_vectors),np.array(encoded_labels),10)