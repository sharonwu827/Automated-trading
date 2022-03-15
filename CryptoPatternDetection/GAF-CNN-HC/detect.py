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
from keras.utils import CustomObjectScope
import click

from data.crypto_compare import *
from tensorflow.keras.models import load_model
import keras_metrics

from datetime import datetime


class ModelPredict(object):

    def __init__(self, target, rule, url_his, url_real, his_ls, real_ls, signal_ls, pattern_ls, save_plot, look_back=7,
                 feature_channels=['open', 'high', 'low', 'close'], model_path ='./model/CNN_8COIN3Ccomb2_USD.h5'):
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
        self.look_back = look_back
        self.columns = feature_channels
        self.pattern_ls = pattern_ls
        for i, j in zip(pattern_ls, range(len(pattern_ls))):
            self.pattern_dict[j] = i
        self.pattern_dict[len(signal_ls)] = 'No Pattern'
        self.package_realtime = None
        self.model_path = model_path
        self.predict = None


    def process(self, feature_channels=None):
        data_1D_pattern = pd.read_csv(self.data_pattern)
        if feature_channels:
            data_1D_pattern = data_1D_pattern[feature_channels]
        sample_size = 24*self.look_back-8
        gasf_arr = np.zeros((sample_size, 10, 10, len(self.columns)))

        data_since_ten = data_1D_pattern.iloc[9:]
        gasf_arr = util_gasf.detect(data_1D_pattern, 'n', columns = self.columns, d=data_since_ten)

        with open(self.gasf_arr, 'wb') as handle:
            pickle.dump(gasf_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_realtime(self):
        with open(self.gasf_arr, 'rb') as handle:
            self.load_data = pickle.load(handle)

    def predict_realtime(self):
        # model = load_model(self.model_path, custom_objects={'precision': Precision,'recall':Recall})
        # model_path = './model/CNN_8COIN3Ccomb2_USD.h5'
        # with CustomObjectScope({'precision': Precision(), 'recall': Recall()}):
        with CustomObjectScope({'precision': keras_metrics.precision(), 'recall': keras_metrics.recall()}):
            self.load_model = load_model(self.model_path)
        print(self.load_model.summary())

        self.predict= self.load_model.predict(self.load_data)
        print(self.predict)
        # prediction = np.argmax(predict_x, axis=1)



@click.command()
@click.option('-mode', default='csv_download', help='csv_download, gasf, gasf+cnn')
@click.option('-targets', default='BTC_USD', help='target coin.')
@click.option('-frequency', default='hour', help='price frequency.')
@click.option('-feature_channels', default="open,high,low,close", help='feature channels')  #['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']
@click.option('-pattern_ls', default="MorningStar_good,MorningStar_bad,EveningStar_good,EveningStar_bad", help='pattern ls')
@click.option('-look_back', default=7, help='days look back for prediction')

def run(mode, targets, frequency, feature_channels, pattern_ls, look_back):
    for target in targets.split(','):
        rule = '1D'
        url_his = None
        url_real = None
        his_ls = ['date', 'open', 'high', 'low', 'close', 'volume']
        real_ls = ['timestamp', 'open', 'dayHigh', 'dayLow', 'price']
        pattern_ls = pattern_ls.split(',')
        signal_ls = ['MorningStar', 'EveningStar']
        save_plot = False
        file_name = f'./temp/{target}_history.csv'
        look_back = look_back
        main = ModelPredict(target, rule, url_his, url_real, his_ls, real_ls, signal_ls, pattern_ls, save_plot, look_back, feature_channels.split(','))
        main.data_pattern = file_name
        main.gasf_arr = './temp/gasf_arr_' + target

        if 'realtime' in mode:
            target = target
            cryptocurrency, target_currency = target.replace('_', '/').split('/')
            ed = int(datetime.now().timestamp())
            data = get_hist_data(cryptocurrency, target_currency, timeframe=frequency, limit=24*look_back,
                          toTs=ed, data_type='price')
            df = data_to_dataframe(data)

            data_social = get_hist_data(cryptocurrency, target_currency, timeframe=frequency, limit=24 * look_back,
                                 toTs=ed, data_type='social')
            df_social = data_to_dataframe(data_social)
            df = df.join(df_social,how='left')
            df.to_csv(file_name)

        if 'process' in mode:
            main.process(feature_channels.split(','))
##

        if 'predict' in mode:
            main.load_realtime()
            main.predict_realtime()

if __name__ == "__main__":
    run()

























