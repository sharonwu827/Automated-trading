## main.py 
Main function for the pipeline of model training.

e.g.

```python
run main.py -mode=csv_download+csv_pattern+gasf+cnn
-targets=BTC_USD
-sample_size=40
-pattern_ls=MorningStar_good,EveningStar_good
-feature_channels=open,high,close,low
-ignore_patternless=False
-start_date=2022-02-01
-end_datedefault=2022-02-15
```
The above run configuraton would load history data of Bitcoin (open,high,close,low prices) between 2022-02-01 and 2022-02-15 as the original data. The dataset would be labeled into categories of MorningStar_good, EveningStar_good and no_pattern by the hardcode method. The CNN model is trained on the dataset that consists of 40 samples for each category, each channel of an individual data sample is a matrix acquired by GAF transformation from the original 1D time series.  

-  _data/_, the package for loading data from cryptocompare.com
- _csv/_, where the original history data and labeled data is stored
- _pattern_images/_, images samples for each category are stored here.
- _gasf_arr/_, where the data is stored after GAF transformation
- _load_data/_, processed data ready for model training
- _model/_, where the trained model is stored



## detect.py

Detect function for realtime prediction on target coin based on the trained model from main.py. 

e.g.

```python
%run detect.py 
-mode=realtime+process+predict+display 
-targets=BTC_USD 
-look_back=7 
-feature_channels=open,high,close,low,volumeto 
```
The above command would do the pattern detection on the Bitcoin history within the most recent 7 days and display its output. The default mode is loaded from model_path ='./model/Buy(MG)_Sell(EG)_No/8COIN3Ccomb2_USD_705_707_701.h5'.- 

 - _temp/_, temp files for each prediction task