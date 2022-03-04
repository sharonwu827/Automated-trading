import os
import glob
import pandas as pd
os.chdir("/mydir")

def csv_combine(filename, coin_ls):
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')



@click.command()
@click.option('-target', default='csv_download', help='target csv file name')
@click.option('-sources', default='BTC_USD,ETH_USD,BNB_USD', help='source csv pattern files')



def run(mode, targets, start_date, end_date, frequency, sample_size, feature_channels):
    for target in targets.split(','):
        rule = '1D'
        url_his = None
        url_real = None
        his_ls = ['date', 'open', 'high', 'low', 'close', 'volume']
        real_ls = ['timestamp', 'open', 'dayHigh', 'dayLow', 'price']
        pattern_ls = ['MorningStar_good', 'MorningStar_bad', 'EveningStar_good', 'EveningStar_bad']
        signal_ls = ['MorningStar', 'EveningStar']
        save_plot = False
        file_name = f'./csv/{target}_history.csv'
        main = PatternModel(target, rule, url_his, url_real, his_ls, real_ls, signal_ls, pattern_ls, save_plot, sample_size, feature_channels.split(','))
        main.gasf_arr = './gasf_arr/gasf_arr_' + target
        main.data_pattern = './csv/' + target + '_pattern.csv'

        if 'csv_download' in mode:
            target = target
            df = get_timeseries_history(target.replace('_', '/'), start_date, end_date, frequency)
            df.to_csv(file_name)
            print(df.head())

        if 'csv_pattern' in mode:
            main.api_history(filename=file_name)
            main.rule_based()

        if 'gasf' in mode: #test
            main.gasf()

        if 'cnn' in mode:
            main.process_xy()
            main.cnn()

if __name__ == "__main__":
    run()