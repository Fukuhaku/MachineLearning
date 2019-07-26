# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:56:38 2019

@author: m_shibata
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


train_file = 'train.csv' ## 学習用データ
test_file = 'test.csv' ## テスト用データ
submit_file = 'titanic_submission.csv' ## 提出用データ
        
test_size = 0.2 ## テスト用データの割合
c = 1.0 ## 正則化のパラメータ（デフォルト 1.0）

## 分類レポートのラベル
targets = ['死亡', '生存']  ## 0 : 死亡, 1 : 生存

drop_cols = {'train' : ['Survived', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin'], 
             'test' : ['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin']}

  
class Titanic:
    ## コンストラクタ
    def __init__(self, train_file, test_file):
        
        ## データを読み込む
        self.df = pd.read_csv(train_file, index_col=0, encoding='UTF-8')
        self.df_test = pd.read_csv(test_file, index_col=None, encoding='UTF-8')
        
        ## 学習用データ・テスト用データ
        self.df_dummies = self.preprocess(self.df)
        self.df_test_dummies = self.preprocess(self.df_test)
        
        ## ロジスティック回帰のモデル
        self.clf = linear_model.LogisticRegression(C=c, solver='liblinear', 
                                                   random_state=0)
        ## 回帰係数
        self.coef = []
        
        
    ## データを確認
    def check_data(self):
        
        ## 学習用データ・テスト用データの基本統計量
        print('学習用データの基本統計量')
        print(self.df.describe(), '\n')
        
        print('テスト用データの基本統計量')
        print(self.df_test.describe(), '\n')
    
    
    ## データ前処理（欠損値・外れ値・One-Hot-Encoding・標準化）
    def preprocess(self, df):
        ## 欠損値の処理（Age, Cabin, Embarked）
        df['Age'] = df['Age'].fillna(df['Age'].median())    ## 中央値
        df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode())    ## 最頻値
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())    ## 最頻値
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())    ## 中央値
        
        ## 外れ値の処理
        ## Parch（両親や子供の数） の値 9 は学習用データになく、テスト用データのみにあるので、
        ## 学習用データの最大値 6 に置換
        df['Parch'] = np.where(df['Parch']==9, 6, df['Parch'])
        
        ## ダミー変数（One-Hot Encoding）
        ## drop_first=True の場合, 基準レベルを除く
        df_dummies = pd.get_dummies(df, columns=['Sex', 'Pclass', 'SibSp', 
                                                 'Parch', 'Embarked'])
        ## データを標準化（平均 0, 分散 1）
        scaler = StandardScaler()
        
        df_dummies['Age_scale'] = scaler.fit_transform(df_dummies.loc[:, ['Age']])
        df_dummies['Fare_scale'] = scaler.fit_transform(df_dummies.loc[:, ['Fare']])
        
        return df_dummies
        
    
    ## 目的変数・説明変数を作成
    def split_data(self, flag):
        
        ## 学習用データの場合
        if flag == 'train':
            ## 説明変数
            X = self.df_dummies.drop(drop_cols[flag], axis=1)
            ## 目的変数
            y = self.df_dummies['Survived'].values
            
            return X, y
        else:
            ## 説明変数
            X = self.df_test_dummies.drop(drop_cols[flag], axis=1)
            
            return X
    
    
    ## 予測モデルを作成
    def model(self, X, y):
        
        self.clf.fit(X, y)
        
        self.coef = self.clf.coef_
     
        
    ## 説明変数の回帰係数を表示
    def show_coef(self, columns):
        n_features = list(columns)
        fti = np.reshape(self.coef, -1).tolist() ## 配列型へ変換
        
        print('説明変数の回帰係数:')
        for i, feature in enumerate(columns):
            print('\t{0:10s} : {1:>.6f}'.format(feature, fti[i]))
        
        ## グラフ化（説明変数の順番を逆に表示）  
        plt.figure(figsize=(4, 6))
        plt.barh(np.arange(len(n_features))[::-1], fti, align='center')
        plt.yticks(np.arange(len(n_features))[::-1], n_features)
        plt.title('説明変数の影響度（回帰係数）')
        plt.xlabel('回帰係数の値')
        plt.ylabel('説明変数')
        plt.grid()
        plt.show()
        
        
    ## 予測
    def predict(self, X):
        ## 予測値を計算
        y = self.clf.predict(X)
        
        return y
    
    
    ## モデルを検証
    def verification(self, y_valid, y_valid_pred):
        ## 混同行列
        confusion = confusion_matrix(y_valid, y_valid_pred)
        print('混同行列（検証用データ）')
        print(confusion, '\n')
    
        print('分類レポート（検証用データ）')
        print(classification_report(y_valid, y_valid_pred, 
                                    target_names=targets))
    
    
    ## 提出用データをエクスポート（id・予測値）
    def export_data(self, predict, submit_file):
        ## 予測結果とスコアをテストデータに追加
        self.df_test['Survived'] = predict
        
        ## 提出用データをエクスポート（id・予測値）
        ## CSV ファイルを出力
        self.df_test.loc[:, ['PassengerId', 'Survived']].to_csv(
                submit_file, index=False, header=True)
    
    
## メイン
def main(titanic):
    
    ## データを確認
    titanic.check_data()
    
    ## 学習用データより目的変数・説明変数を作成
    (X, y) = titanic.split_data('train')
    
    ## 学習用・検証用データに分割
    (X_train, X_valid, y_train, y_valid) = train_test_split(
            X, y, test_size=test_size, random_state=0)
    
    ## 予測モデルを作成
    titanic.model(X_train, y_train)
    
    ## 説明変数の回帰係数を表示
    titanic.show_coef(X.columns)
    
    ## 予測
    y_valid_pred = titanic.predict(X_valid)
    
    ## モデルを検証
    titanic.verification(y_valid, y_valid_pred)
    
    ## テスト用データを予測
    predict_result = titanic.predict(titanic.split_data('test'))
    
    ## 予測結果をエクスポート
    titanic.export_data(predict_result, submit_file)


if __name__ == '__main__':
    
    titanic = Titanic(train_file, test_file) 
    
    main(titanic)
 