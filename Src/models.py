import preprocessing
import utils
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

sentiment_dicts =  preprocessing.convert_lex_to_dict()

tweets,labels = utils.load_dataset('../twitter_data/train2017.tsv')

encoded_labels = preprocessing.one_hot_encode(labels)

list_of_processed_tweets = preprocessing.process_tweets(tweets)

sentiment_vectors = preprocessing.create_sentiment_vectors(list_of_processed_tweets,sentiment_dicts)
vectorized_tweets = preprocessing.tfidf_vectorization(list_of_processed_tweets)
dim_reduced_tweets = preprocessing.dim_reduction(vectorized_tweets)

final_vectors = preprocessing.concatenate_vectors(dim_reduced_tweets,sentiment_vectors)

clf = KNeighborsClassifier(n_neighbors=5)

print("10FoldCross Validation...")
utils.KfoldCrossValidation(clf,np.array(final_vectors),np.array(encoded_labels),10)


