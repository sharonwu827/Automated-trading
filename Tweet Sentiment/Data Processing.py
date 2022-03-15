import random
import time
import datetime
import numpy as np
# Set up TWINT config
import twint
# Solve compatibility issues with notebooks and RunTime errors.
import nest_asyncio
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


def tweets_api(keyword, Language, min_replies):
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
    c.Since = '2019-03-16 14:00:00'
    c.until = '2022-03-01 06:00:00'
    c.Min_replies = min_replies
    c.Pandas = True
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    df = df[["date", "username", "tweet", "hashtags", "nreplies", "nretweets", "nlikes"]]
    return df


def text_preprocessing(tweet):
    tweet = re.sub(text_cleaning_regex, ' ', str(tweet).lower()).strip()
    res=[]
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

df['sentiment_dict'] = df['tweet'].apply(lambda x:analyzer.polarity_scores(x))
df['compound']  = df['sentiment_dict'].apply(lambda score_dict: score_dict['compound'])
df['neg']  = df['sentiment_dict'].apply(lambda score_dict: score_dict['neg'])*-1
df['pos']  = df['sentiment_dict'].apply(lambda score_dict: score_dict['pos'])

#  Prediction confidence
df['confidence score'] = abs(df['neg']+df['pos'])/(abs(df['neg'])+abs(df['pos']))
df['confidence score'] = df['confidence score'].replace(np.nan, 0)








def twint_loop(searchterm, since, until):
    '''

    :param searchterm: keywork searched
    :param since:
    :param until:
    :return: df
    '''

    def twint_search(searchterm, since, until, json_name):
        '''
        Twint search for a specific date range.
        Stores results to df.
        '''
        nest_asyncio.apply()
        c = twint.Config()
        c.Search = searchterm
        c.Since = since
        c.Until = until
        c.Hide_output = True
        c.Pandas = True
        c.Output = json_name

        try:
            twint.run.Search(c)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print("Problem with %s." % since)

    daterange = pd.date_range(since, until)

    for start_date in daterange:

        since= start_date.strftime("%Y-%m-%d")
        until = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")

        json_name = '%s.json' % since
        json_name = path.join(dirname, json_name)

        print('Getting %s ' % since )
        twint_search(searchterm, since, until, json_name)

