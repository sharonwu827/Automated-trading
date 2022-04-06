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

    def __init__(self, target, rule, url_his, url_real, his_ls, real_ls, signal_ls, pattern_ls, save_plot, filter_model, look_back=7,
                 feature_channels=['open', 'high', 'low', 'close'], model_path ='./model/Buy(MG)_Sell(EG)_No/8COIN3Ccomb2_USD_705_707_701.h5',
                 extra_feature_channels = ['open', 'high', 'low', 'close'], extra_model_path='./model/Morning_Evening_No/CNN_8COINMEN4F_USD_906_906_906.h5'):
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
        self.predict_result = None
        self.filter_model = filter_model
        self.extra_model_path = extra_model_path
        self.extra_load_model = None
        self.extra_columns = extra_feature_channels
        self.extra_gasf_arr = None
        self.extra_load_data = None
        self.extra_predict = None
        self.check_pattern = None



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

        if self.filter_model:
            data_1D_pattern = pd.read_csv(self.data_pattern)
            if self.extra_columns:
                data_1D_pattern = data_1D_pattern[self.extra_columns]
            sample_size = 24 * self.look_back - 8
            gasf_arr_extra = np.zeros((sample_size, 10, 10, len(self.extra_columns)))

            data_since_ten = data_1D_pattern.iloc[9:]
            gasf_arr_extra = util_gasf.detect(data_1D_pattern, 'n', columns=self.extra_columns, d=data_since_ten)

            with open(self.extra_gasf_arr, 'wb') as handle:
                pickle.dump(gasf_arr_extra, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def load_realtime(self):
        with open(self.gasf_arr, 'rb') as handle:
            self.load_data = pickle.load(handle)

        if self.filter_model:
            with open(self.extra_gasf_arr, 'rb') as handle:
                self.extra_load_data = pickle.load(handle)

    def predict_realtime(self, feature_channels=None):
        # model = load_model(self.model_path, custom_objects={'precision': Precision,'recall':Recall})
        # model_path = './model/CNN_8COIN3Ccomb2_USD.h5'
        # with CustomObjectScope({'precision': Precision(), 'recall': Recall()}):
        with CustomObjectScope({'precision': keras_metrics.precision(), 'recall': keras_metrics.recall()}):
            self.load_model = load_model(self.model_path)
        # print(self.load_model.summary())

        prediction= self.load_model.predict(self.load_data)
        self.predict = np.argmax(prediction, axis = 1)
        na_filling = np.empty(9)
        na_filling[:] = np.nan
        self.predict = np.append(na_filling, self.predict)
        # print(self.predict)
        # prediction = np.argmax(predict_x, axis=1)
        
        df = pd.read_csv(self.data_pattern)
        df = df.set_index('time')
        df['predict'] = self.predict
        df = df[feature_channels+['predict']]


        if self.filter_model:
            with CustomObjectScope({'precision': keras_metrics.precision(), 'recall': keras_metrics.recall()}):
                self.extra_load_model = load_model(self.extra_model_path)

            prediction = self.extra_load_model.predict(self.extra_load_data)
            self.extra_predict = np.argmax(prediction, axis=1)
            na_filling = np.empty(9)
            na_filling[:] = np.nan
            self.extra_predict = np.append(na_filling, self.extra_predict)
            df['extra_predict'] = self.extra_predict

            def filter_predict(p, pe):
                if p == 0 and pe == 0:
                    return 0
                elif p == 1 and pe == 1:
                    return 1
                else:
                    return 2
            df['predict'] = df.apply(lambda x: filter_predict(x.predict, x.extra_predict), axis=1)

        df.to_csv(self.predict_result)

    def display(self):
        df = pd.read_csv(self.predict_result)
        df['datetime'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df = df.set_index('datetime')

        plt.figure(figsize=(20, 5))
        plt.plot(df.close, label='Close')

        plt.scatter(df[df['predict'] == 1].index, df[df['predict'] == 1].close, label="SELL", c='red')
        plt.scatter(df[df['predict'] == 0].index, df[df['predict'] == 0].close, label="BUY", c='green', marker='x')

        plt.legend()
        plt.show()

        print("Buy:")
        print(df[df['predict'] == 0].close)

        print("Sell:")
        print(df[df['predict'] == 1].close)



    def backcheck(self):
        Sig = Signal(self.predict_result, self.signal_ls, self.save_plot, self.pattern_ls)
        Sig.process()
        self.check_pattern = Sig.back_detect_all(self.target, look_forward=8)


@click.command()
@click.option('-mode', default='realtime', help='model chose for predict')
@click.option('-targets', default='BTC_USD', help='target coin.')
@click.option('-frequency', default='hour', help='price frequency.')
@click.option('-feature_channels', default="open,high,close,low,volumeto,reddit_active_users,reddit_posts_per_hour,posts,total_page_views", help='feature channels')
@click.option('-pattern_ls', default="MorningStar_good,MorningStar_bad,EveningStar_good,EveningStar_bad", help='pattern ls')
@click.option('-look_back', default=7, help='days look back for prediction')
@click.option('-filter_model', default=True, help='if include the extra 4f model for morning/evening classifier')

def run(mode, targets, frequency, feature_channels, pattern_ls, look_back, filter_model):
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
        filter_model = filter_model
        main = ModelPredict(target, rule, url_his, url_real, his_ls, real_ls, signal_ls, pattern_ls, save_plot, filter_model, look_back, feature_channels.split(','))
        main.data_pattern = file_name
        main.gasf_arr = './temp/gasf_arr_' + target
        main.predict_result = f'./temp/{target}_result.csv'
        main.extra_gasf_arr = './temp/extra_gasf_arr_' + target

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
            main.predict_realtime(feature_channels.split(','))

        if 'display' in mode:
            main.display()

        if 'backcheck' in mode:
            main.backcheck()

if __name__ == "__main__":
    run()

























