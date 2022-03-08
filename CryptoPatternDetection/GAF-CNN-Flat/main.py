from Api_history import Api_history
from Signal2 import Signal
import util_gasf
from Cnn import CNN
from Api_realtime import Api_realtime
import util_pattern
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
from keras.models import load_model
from data.crypto_compare import *

import click


class PatternModel(object):

    def __init__(self, target, rule, url_his, url_real, his_ls, real_ls, signal_ls, pattern_ls, save_plot, sample_size=30,
                 feature_channels=['open', 'high', 'low', 'close']):
        self.target = target
        self.rule = rule
        self.url_his = url_his
        self.url_real = url_real
        self.his_ls = his_ls
        self.real_ls = real_ls
        self.signal_ls = signal_ls
        self.save_plot = save_plot
        self.data = None
        self.data_pattern = None
        self.gasf_arr = None
        self.load_data = None
        self.load_model = None
        self.pattern_dict = dict()
        self.sample_size = sample_size
        self.columns = feature_channels
        self.pattern_ls = pattern_ls
        for i, j in zip(pattern_ls, range(len(pattern_ls))):
            self.pattern_dict[j] = i
        self.pattern_dict[len(signal_ls)] = 'No Pattern'
        self.package_realtime = None

    def api_history(self, filename):
        # api_his = Api_history(self.url_his, self.his_ls, self.target, self.rule)
        self.data = filename
        # api_his.summary_history()

    def rule_based(self):
        Sig = Signal(self.data, self.signal_ls, self.save_plot, self.pattern_ls)
        Sig.process()
        self.data_pattern = Sig.detect_all(self.target, look_forward=8)
        Sig.summary()

    def gasf(self):
        data_1D_pattern = pd.read_csv(self.data_pattern)
        gasf_arr = np.zeros((len(self.pattern_ls) + 1, self.sample_size, 10, 10, len(self.columns))) # choose 30 samples for each signal
        for i, j in zip(self.pattern_ls, range(len(self.pattern_ls))):
            gasf = util_gasf.detect(data_1D_pattern, i, columns=self.columns)
            if gasf.shape[0] == 0:  # if there is no sample for signal
                continue  # ingore for now
            gasf_arr[j, :, :, :, :] = gasf[0:self.sample_size, :, :, :]  # data balancing
        df = data_1D_pattern.copy()
        for i in self.pattern_ls:
            df = df.loc[df[i] == 0]  # none signal pattern
        df = shuffle(df[9::])  # 1 out of 9
        gasf = util_gasf.detect(data_1D_pattern, 'n', columns=self.columns, d=df)
        gasf_arr[-1, :, :, :, :] = gasf[0:self.sample_size, :, :, :]
        # self.gasf_arr = './gasf_arr/gasf_arr_' + self.target
        with open(self.gasf_arr, 'wb') as handle:
            pickle.dump(gasf_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process_xy(self):
        with open(self.gasf_arr, 'rb') as handle:
            gasf_arr = pickle.load(handle)
        x_arr = np.zeros(((len(self.pattern_ls) + 1), self.sample_size, 10, 10, len(self.columns)))
        for i in range(len(self.pattern_ls) + 1):
            x_arr[i, :, :, :, :] = gasf_arr[i, 0:self.sample_size, :, :, :]
        x_arr = gasf_arr.reshape((len(self.pattern_ls) + 1) * self.sample_size, 10, 10, len(self.columns))
        y_arr = []
        for i in range(len(self.pattern_ls) + 1):
            ls = [i] * self.sample_size
            y_arr.extend(ls)
        y_arr = np.array(y_arr)
        load_data = {'data': x_arr, 'target': y_arr}
        self.load_data = './load_data/load_data_' + self.target
        with open(self.load_data, 'wb') as handle:
            pickle.dump(load_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def cnn(self):
        with open(self.load_data, 'rb') as handle:
            load_data = pickle.load(handle)
        x_arr = load_data['data']
        y_arr = load_data['target']
        model = CNN(x_arr, y_arr)
        model.process()
        model.build()
        model.train(0.2)
        model.show()
        self.load_model = './model/CNN_' + self.target + '.h5'
        model.save(self.load_model)

    def api_realtime(self):
        api_real = Api_realtime(self.url_his, self.url_real, self.his_ls, self.real_ls)
        self.package_realtime = api_real.detect_real()

    def predict_realtime(self):
        model = load_model(self.load_model)
        if self.package_realtime != False:
            df_real, period = self.package_realtime[0], self.package_realtime[1]
            df = shuffle(df_real.iloc[9::])
            gasf = util_gasf.detect(df_real, 'n', df)
            x_realtime_arr = gasf
            y_pred_realtime = model.model.predict_classes(x_realtime_arr)
            pattern = self.pattern_dict[y_pred_realtime[0]]
            print('Target: {}'.format(self.target))
            print('Time Rule: {}'.format(self.rule))
            print('Time Period: {} - {}'.format(period[0], period[1]))
            print('The Pattern of the Realtime Data: {}'.format(pattern))
            util_pattern(df_real, self.rule, pattern)
            return (period, pattern)
        else:
            print('Not in the transaction time')
            return False


@click.command()
@click.option('-mode', default='csv_download', help='csv_download, gasf, gasf+cnn')
@click.option('-targets', default='BTC_USD', help='Number of greetings.')
@click.option('-start_date', default='2022-01-01', help='Number of greetings.')
@click.option('-end_date', default='2022-02-15', help='Number of greetings.')
@click.option('-frequency', default='hour', help='Number of greetings.')
@click.option('-sample_size', default=30, help='Number of cvs samples to gasf.')
@click.option('-feature_channels', default="open,high,low,close", help='feature channels')  #['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']

def run(mode, targets, start_date, end_date, frequency, sample_size, feature_channels):
    for target in targets.split(','):
        rule = '1D'
        url_his = None
        url_real = None
        his_ls = ['date', 'open', 'high', 'low', 'close', 'volume']
        real_ls = ['timestamp', 'open', 'dayHigh', 'dayLow', 'price']
        pattern_ls = ['MorningStar_good', 'MorningStar_bad', 'EveningStar_good', 'EveningStar_bad'] # ['MorningStar_good', 'MorningStar_bad', 'EveningStar_good', 'EveningStar_bad']
        signal_ls = ['MorningStar', 'EveningStar']
        save_plot = True
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




























