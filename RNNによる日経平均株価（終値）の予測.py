# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:21:22 2019

@author: m_shibata

本プログラムを参考にした取引によって発生した損害、またはトラブルについては、
一切の責任を負いかねます。
"""

## RNN による日経平均株価（終値）の予測
## 過去30日分の株価より当日の株価を予測

## 過去300～61日分を訓練用データ
## 過去60～31日分を検証用データ
## 過去30～0日分をテスト用データ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

## 以下の URL より日経平均株価データ（日別）をダウンロードし、
## このプログラムと同一階層に保存
## https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_jp.csv

data_file = 'nikkei_stock_average_daily_jp.csv'

## 当日の株価を予測するために必要な過去の日数
lookback = 30

## エポック数
epochs = 3000

## データファイルを読み込む
## データ日付を index に設定
## 最終行はデータではないため、スキップ
df = pd.read_csv(data_file, index_col=0, encoding='cp932', 
                 skipfooter=1, engine='python')

## 終値
closing_price = df[['終値']].values

## 訓練・検証・テスト用データを作成
## 過去30日分の株価より当日の株価とする
def data_split(data, start, end, lookback):
    length = abs(start-end)
    
    X = np.zeros((length, lookback))
    y = np.zeros((length, 1))
    
    for i in range(length):
        j = start - lookback + i
        k = j + lookback
        
        X[i] = data[j:k, 0]
        y[i] = data[k, 0]
        
    return X, y

## 訓練・検証・テスト用データ
(X_train, y_train) = data_split(closing_price, -300, -60, lookback)
(X_valid, y_valid) = data_split(closing_price, -60, -30, lookback)
(X_test, y_test) = data_split(closing_price, -30, 0, lookback)

## 標準化
## X のみ次元を変換（2次元 ⇒ 3次元）
scaler = StandardScaler()

scaler.fit(X_train)
X_train_std = scaler.transform(X_train).reshape(-1, lookback, 1)
X_valid_std = scaler.transform(X_valid).reshape(-1, lookback, 1)
X_test_std = scaler.transform(X_test).reshape(-1, lookback, 1)

scaler.fit(y_train)
y_train_std = scaler.transform(y_train)
y_valid_std = scaler.transform(y_valid)

## 訓練 RNN
model = Sequential()
model.add(GRU(128, 
              dropout=0.2, 
              recurrent_dropout=0.2, 
              return_sequences=False, 
              input_shape=(None, X_train_std.shape[-1])))
model.add(Dense(1))

model.compile(optimizer=RMSprop(), loss='mae', metrics=['accuracy'])

result = model.fit(X_train_std, y_train_std, 
                   verbose=0,   ## 詳細表示モード
                   epochs=epochs, 
                   batch_size=64, 
                   shuffle=True, 
                   validation_data=(X_valid_std, y_valid_std))

## 訓練の損失値をプロット
epochs = range(len(result.history['loss']))
plt.title('損失値（Loss）')
plt.plot(epochs, result.history['loss'], 'bo', alpha=0.6, marker='.', label='訓練', linewidth=1)
plt.plot(epochs, result.history['val_loss'], 'r', alpha=0.6, label='検証', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

## 予測値
df_predict_std =  pd.DataFrame(model.predict(X_test_std), columns=['予測値'])

## 予測値を元に戻す
predict = scaler.inverse_transform(df_predict_std['予測値'].values)

## 予測結果をプロット
pre_date = df.index[-len(y_test):].values
plt.title('実際の終値と予測値')
plt.plot(pre_date, y_test, 'b', alpha=0.6, marker='.', label='実際の終値', linewidth=1)
plt.plot(pre_date, predict, 'r', alpha=0.6, marker='.', label='予測値', linewidth=1)
plt.xticks(rotation=70)
plt.legend()
plt.grid(True)
plt.show()

## RMSEの計算
print('二乗平均平方根誤差（RMSE） : %.3f' %  
       np.sqrt(mean_squared_error(y_test, predict)))
