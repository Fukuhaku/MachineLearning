# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:07:02 2019

@author: m_shibata
"""

import pandas as pd

train_file = 'train.csv' ## 学習用データ
test_file = 'test.csv' ## テスト用データ

class Titanic:
    ## コンストラクタ
    def __init__(self, train_file, test_file):
        
        ## データを読み込む
        self.df = pd.read_csv(train_file, index_col=0, encoding='UTF-8')
        self.df_test = pd.read_csv(test_file, index_col=None, encoding='UTF-8')
        
    ## データを確認
    def check_data(self):
        
        ## 学習用データ・テスト用データの基本統計量
        print('学習用データの基本統計量')
        print(self.df.describe(), '\n')
        
        print('テスト用データの基本統計量')
        print(self.df_test.describe(), '\n')
    
## メイン
def main(titanic):
    
    ## データを確認
    titanic.check_data()

if __name__ == '__main__':
    
    titanic = Titanic(train_file, test_file) 
    
    main(titanic)
 