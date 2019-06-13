# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:29:27 2019

@author: m_shibata
"""

## 本プログラムを基に発生した損失等に関しては、私は責任を負いかねます。

## NN による日経平均株価（終値）の予測
## 過去30日分の株価を説明変数、当日の株価を目的変数とする

## 過去300～61日分を訓練データ
## 過去60～31日分を検証データ
## 過去30～0日分をテストデータ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
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
df = pd.read_csv(data_file, index_col=0, encoding='cp932')

## 最終行はデータではないため、削除
df = df[:-1]

## 終値
closing_price = df['終値'].values

## 欠損値の確認
print('欠損値の個数')
print(df.isnull().sum(), '\n')

## 基本統計量の確認（終値）
print('終値の基本統計量')
print(df['終値'].describe(), '\n')

## 終値を時系列にプロット
plt.title('日経平均株価（終値）の推移')
plt.plot(range(len(closing_price)), closing_price)
plt.show()

plt.title('日経平均株価（終値）の推移　過去300日分のみ')
plt.plot(range(len(closing_price[-300:])), closing_price[-300:])
plt.show()

plt.title('日経平均株価（終値）の推移　過去30日分のみ')
plt.plot(df.index[-30:], closing_price[-30:])
plt.xticks(rotation=70)
plt.show()

## 訓練・検証・テスト用データを作成
## 過去30日分の株価より当日の株価とする
def data_split(data, start, end, lookback):
    length = abs(start-end)
    X = np.zeros((length, lookback))
    y = np.zeros(length,)
    
    if end == 0:
        ## テスト用データの場合
        start_temp = start - lookback
    else:
        start_temp = start
           
    for i in range(length):
        j = start_temp + i
        k = j + lookback
        
        X[i] = data[j:k]
        y[i] = data[k]
        
    return X, y

## 訓練・検証・テスト用データ
(X_train, y_train) = data_split(closing_price, -300, -60, lookback)
(X_valid, y_valid) = data_split(closing_price, -60, -30, lookback)
(X_test, y_test) = data_split(closing_price, -30, 0, lookback)

## 訓練
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[-1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=RMSprop(), loss='mae', metrics=['accuracy'])

result = model.fit(X_train, y_train, 
                   verbose=0,   ## 詳細表示モード
                   epochs=epochs, 
                   batch_size=64, 
                   validation_data=(X_valid, y_valid))

## 訓練の損失値をプロット
epochs = range(len(result.history['loss']))
plt.title('損失値（Loss）')
plt.plot(epochs, result.history['loss'], 'bo', alpha=0.6, marker='.', label='訓練', linewidth=1)
plt.plot(epochs, result.history['val_loss'], 'r', alpha=0.6, label='検証', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 2000)
plt.show()

## 予測値
df_predict =  pd.DataFrame(model.predict(X_test), columns=['予測値'])

## 予測結果をプロット
#epochs = range(len(y_test))
epochs = df.index[-30:]
plt.title('実際の終値と予測値')
plt.plot(epochs, y_test, 'b', alpha=0.6, marker='.', label='実際の終値', linewidth=1)
plt.plot(epochs, df_predict['予測値'].values, 'r', alpha=0.6, marker='.', label='予測値', linewidth=1)
plt.xticks(rotation=70)
plt.legend()
plt.grid(True)
plt.show()

## RMSEの計算
print('二乗平均平方根誤差（RMSE） : %.3f' %  
       np.sqrt(mean_squared_error(y_test, df_predict['予測値'].values)))
