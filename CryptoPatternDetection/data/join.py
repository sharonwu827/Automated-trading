import pandas as pd
import click

@click.command()
@click.option('-coins', default='ETH', help='csv file for bitcoin price history')
# @click.option('-target_file', default='ETHnews_USD_history.csv', help='csv file for bitcoin price history')

def run(coins):
    df = pd.read_csv('./news_history.csv')
    df = df[['sentiment', 'rank_score', 'datetime', 'sentimenttitle', 'sentimenttext']]

    for coin in coins.split(','):
        history_file = './csv/'+coin+'_USD_history.csv'
        df_0 = pd.read_csv(history_file)
        df_0 = df_0[['time', 'high', 'low', 'open', 'volumeto', 'close', 'reddit_active_users', 'reddit_posts_per_hour', 'posts', 'total_page_views']]

        df_join = df_0.set_index('time').join(df.set_index('datetime'))

        first_row = df_join[df_join['sentiment'].notna()].index[0]
        last_row = df_join[df_join['sentiment'].notna()].index[-1]

        df_filter = df_join[(df_join.index >= first_row) & (df_join.index <=last_row)]
        sparse_rate = df_filter['sentiment'].isna().sum()/df_filter.shape[0]
        print(df_filter.shape)
        print('Sparse rate:', sparse_rate)

        df_filter = df_filter.fillna(method="ffill")
        df_filter.index.name = 'time'

        target_file = coin+'news_USD_history.csv'
        df_filter.to_csv(target_file)

if __name__ == "__main__":
    run()