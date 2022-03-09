import random
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
% matplotlib
inline
import sys
import os
# Set up TWINT config
import twint
# Solve compatibility issues with notebooks and RunTime errors.
import nest_asyncio


def tweets_api(keyword, Language, min_replies, start_time, end_time):
    '''
    :param keyword:
    :param Language:
    :param min_replies:
    :return: processed dataframe
    '''
    nest_asyncio.apply()
    c = twint.Config()
    c.Search = keyword
    c.Lang = Language
    c.Store_pandas = True
    # c.Store_csv = True
    # c.Output = "test.csv"
    c.Since = start_time
    c.until = end_time
    c.Min_replies = min_replies
    c.Pandas = True
    c.Hide_output = True
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    df = df[["date", "username", "tweet", "hashtags", "nreplies", "nretweets", "nlikes"]]


def text_preprocessing(tweet):
    tweet = re.sub(text_cleaning_regex, ' ', str(tweet).lower()).strip()
    res = []
    # Lowercase
    tweet = tweet.lower()
    # Remove single letter words
    # tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    # Remove stopword
    tweet = ' '.join([word for word in tweet.split() if word.lower() not in sw_nltk])
    # remove the word after @ OR #
    for i in tweet.split():
        if i.startswith("@") or i.startswith("#"):
            continue
        else:
            res.append(i)
    return ' '.join(res)


def vader_sentiment_result(sent):
    scores = analyzer.polarity_scores(sent)
    if scores["compound"] >=0.05:
        return 2
    elif scores["compound"] <=-0.05:
        return 0
    else:
        return 1



# df['n_words'] = df['tweet'].apply(lambda x: len(x.split()))
df['tweet']=df['tweet'].apply(lambda x: text_preprocessing(x))
df["vader_result"] = df["tweet"].apply(lambda x: vader_sentiment_result(x))

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df['sentiment_dict'] = df['tweet'].apply(lambda x:analyzer.polarity_scores(x))
df['compound']  = df['sentiment_dict'].apply(lambda score_dict: score_dict['compound'])
df['neg']  = df['sentiment_dict'].apply(lambda score_dict: score_dict['neg'])*-1
df['pos']  = df['sentiment_dict'].apply(lambda score_dict: score_dict['pos'])

#  Prediction confidence
df['confidence score'] = abs(df['neg']+df['pos'])/(abs(df['neg'])+abs(df['pos']))
df['confidence score'] = df['confidence score'].replace(np.nan, 0)








