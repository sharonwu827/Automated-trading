import pandas as pd
import click

@click.command()
@click.option('-history_file', default='ETHUSD_history.csv', help='csv file for bitcoin price history')
@click.option('-target_file', default='ETHnewsUSD_history.csv', help='csv file for bitcoin price history')

def run(history_file, target_file):
    df_0 = pd.read_csv(history_file)
    df = pd.read_csv('./news_history.csv')

    df = df[['sentiment', 'rank_score', 'datetime']]



if __name__ == "__main__":
    run()