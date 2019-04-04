import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import math
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import  WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# create a list of dictionaries, containing one dictionary for each available lexicon
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

# creating a vector capturing tweets's sentiment behaviour
# the vector contains:
## the mean valance of tweet using each one of the available lexicons (5 slots equal to lexs number)
## the total valance using all the lexs of the first and the second half of the tweet
## the min and max valance of any word in the tweet using all the lexs
## tweet's number of words
def create_sentiment_vectors(tweets,dicts):

	vectors = []

	for t in tweets:
		tweet_array = np.zeros(len(dicts))
		extra_tweet_array = np.zeros(5)
		extra_tweet_array[4] = len(t)
		minv = 1000
		maxv = -1000
		median_index = len(t)/2
		for i in range(0,len(dicts)):
			count = 0
			for word in t:
				value = dicts[i].get(word)
				if(value!=None):
					count += 1
					if(count < median_index):
						extra_tweet_array[2] += value
					elif(count >= median_index):
						extra_tweet_array[3] += value
					if(value < minv):
						extra_tweet_array[0] = value
						minv = value
					if(value > maxv):
						extra_tweet_array[1] = value
						maxv = value
					tweet_array[i] += value
			#compute the mean valance
			if(count!=0):
				tweet_array[i] = tweet_array[i]/float(count)

		vectors.append(np.concatenate((tweet_array,extra_tweet_array),axis=0))

	return np.array(vectors)

#exclude non alphabetic words, tokenization, stemming and stopwords removal
def process_tweets(tweets):

	stop_words = set(stopwords.words('english')) 

	count = 0

	list_of_tokenized_tweets = []
	for t in tweets:
		##tokenize
		#tknzr = TweetTokenizer()
		tknzr = RegexpTokenizer(r'\w+')
		tokenized_tweet = tknzr.tokenize(t)
		##convert to lower case
		tokenized_tweet = [w.lower() for w in tokenized_tweet]
		##exclude non alphabetic words
		only_alpha_sentence = [word for word in tokenized_tweet if word.isalpha()]
		##remove stopwords
		filtered_sentence = [w for w in only_alpha_sentence if not w in stop_words]
		tweet_list = []
		##stemming or lemmatization
		#ps = PorterStemmer()
		lemmatizer = WordNetLemmatizer()
		for word in filtered_sentence:
			#tweet_list.append(ps.stem(word))
			tweet_list.append(lemmatizer.lemmatize(word))
		count += 1
		if(count<20):
			print(str(count) + ' ' + str(tweet_list))
		list_of_tokenized_tweets.append(tweet_list)

	return list_of_tokenized_tweets

# exclude links and @,# type hashtags
def exclude_words(tweets):

	res = []
	count = 0
	for t in tweets:
		count += 1
		t_cleaned = ' '.join(item for item in t.split() if not (item.startswith('http')))
		t_cleaned = ' '.join(item for item in t_cleaned.split() if not (item.startswith('@')))
		t_cleaned = ' '.join(item for item in t_cleaned.split() if not (item.startswith('#')))
		res.append(t_cleaned)
		if (count<20):
			print(str(count) + ' ' + t)
			print(t_cleaned)
	return res

# count or tfidf vectorization
def tfidf_vectorization(tweets):

	corpus = []
	for t in tweets:
		string_tweet = ' '
		for w in t:
			string_tweet = string_tweet + w + ' '
		corpus.append(string_tweet)

	vectorizer = TfidfVectorizer()
	#vectorizer = CountVectorizer()
	return vectorizer.fit_transform(corpus)

# dim reduction using trucated svd
def dim_reduction(vectors):
	svd = TruncatedSVD(n_components=400, n_iter=10)
	return svd.fit_transform(vectors)

# concatanate tweet and sentiment vectors
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

# convert labels to nums
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