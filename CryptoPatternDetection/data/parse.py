import pandas as pd

month_dict = dict(Jan='01', Feb='02', Mar='03', Apr='04', May='05', Jun='06', Jul='07', Aug='08', Sep='09', Oct='10',
                  Nov='11', Dec='12')
df = pd.read_csv('./csv/news_sentiment.csv')

df.date = df.date.map(lambda x: x.split(', ')[1])

def replace_month(x):
    try:
        month_code = x.split(' ')[1]
        return x.replace(month_code, month_dict[month_code])
    except:
        print('error')

df.date = df.date.map(lambda x: replace_month(x))
df['offset'] = df.date.map(lambda x: int((x.split(' ')[-1]).replace('0','')))
#df.date = df.date.map(lambda x: ' '.join(x.split(' ')[:-1]))
df.date = pd.to_datetime(df.date.map(lambda x: ' '.join(x.split(' ')[:-1])))
df['datetime'] = df.apply(lambda x: x.date + pd.DateOffset(hours = x.offset), axis=1)
df = df[['datetime','sentiment','rank_score','sentimenttitle','sentimenttext' ]]
df['date']=df.datetime.map(lambda x: x.strftime('%Y-%m-%d'))
df['hour']=df.datetime.map(lambda x: x.hour)
df_grouped =df.groupby(['date','hour']).agg('mean').reset_index()
df_grouped['datetime']= df_grouped.apply(lambda x: '{} {}'.format(x.date, str(x.hour)+':00:00'), axis=1)

df_grouped.to_csv('./news_history.csv')
print('done')