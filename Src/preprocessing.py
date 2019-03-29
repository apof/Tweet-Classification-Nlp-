import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import math
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def convert_lex_to_dict():

	lex_names = ['../lexica/affin/affin.txt','../lexica/emotweet/valence_tweet.txt','../lexica/generic/generic.txt','../lexica/nrc/val.txt','../lexica/nrctag/val.txt']
	list_of_dicts = []

	for name in lex_names:
		sentiment_dict = {}
		with open(name) as fp:  
			line = fp.readline()
			splitted_line = line.split()
			if(len(splitted_line) == 2):
				sentiment_dict[str(splitted_line[0])] = float(splitted_line[1])
			while line:
				line = fp.readline()
				splitted_line = line.split()
				if(len(splitted_line) == 2):
					sentiment_dict[str(splitted_line[0])] = float(splitted_line[1])
		list_of_dicts.append(sentiment_dict)

	return list_of_dicts

def create_sentiment_vectors(tweets,dicts):

	vectors = []

	for t in tweets:
		tweet_array = np.zeros(len(dicts))
		for i in range(0,len(dicts)):
			for word in t:
				value = dicts[i].get("word")
				if(value!=None):
					tweet_array[i] += value
		vectors.append(tweet_array)

	return np.array(vectors)

#tokenization - stemming and stopwords removal
def process_tweets(tweets):

	stop_words = set(stopwords.words('english')) 

	list_of_tokenized_tweets = []
	for t in tweets:
		tknzr = TweetTokenizer()
		tokenized_tweet = tknzr.tokenize(t)
		filtered_sentence = [w for w in tokenized_tweet if not w in stop_words]
		tweet_list = []
		ps = PorterStemmer()
		for word in filtered_sentence:
			tweet_list.append(ps.stem(word))
		list_of_tokenized_tweets.append(tweet_list)

	return list_of_tokenized_tweets

def tfidf_vectorization(tweets):

	corpus = []
	for t in tweets:
		string_tweet = ' '
		for w in t:
			string_tweet = string_tweet + w + ' '
		corpus.append(string_tweet)

	vectorizer = TfidfVectorizer()
	return vectorizer.fit_transform(corpus)

def dim_reduction(vectors):
	svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)
	return svd.fit_transform(vectors)

def concatenate_vectors(vectors,sentiments):

	result_vecs = []

	for i in range(0,len(vectors)):
		vec = []
		for v in vectors[i]:
			vec.append(v)
		for v in sentiments[i]:
			vec.append(v)
		result_vecs.append(np.array(vec))

	return np.array(result_vecs)

def one_hot_encode(labels):

	res = []

	for l in labels:
		if(l=='negative'):
			res.append(0)
		elif(l=='positive'):
			res.append(2)
		else:
			res.append(1)

	return res










