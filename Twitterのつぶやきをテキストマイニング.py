# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 07:26:51 2019

@author: m_shibata
"""

## Twitter のつぶやきデータを Twitter API を利用して取得し、
## テキストマイニング（WordCloud）によって可視化

import json
import pandas as pd
import matplotlib.pyplot as plt
from requests_oauthlib import OAuth1Session
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud

## アクセストークン情報
consumer_api_key = 'xxxxxxxxxxxxxxxx'
consumer_api_secret = 'xxxxxxxxxxxxxxxx'
access_token = 'xxxxxxxxxxxxxxxx'
access_token_secret = 'xxxxxxxxxxxxxxxx'

## API 用 URL
twitter_url = 'https://api.twitter.com/1.1/search/tweets.json?tweet_mode=extended'

## 取得ツイート数（取得ツイート数 100件/1回, 最大リクエスト回数 180回/15分）
count = 100

## 検索語
search_word = 'ラーメン'

## 検索対象エリア（佐賀県庁を中心に半径10km）
search_area = '33.249559,130.299601,10.0km'

## テキストマイニングの対象品詞リスト
word_class = ['名詞', '形容詞']

## 除外する単語リスト
ignore_words = ['こと', 'よう', 'そう', 'これ', 'それ', 'もの', 'ここ', 'さん', 
                'ところ', 'とこ', 'https', 'co', search_word]

## WordCloud 結果ファイル
wordcloud_image_file = './wordcloud.png'

## 日本語フォント
japanese_font = './IPAexfont00301/ipaexg.ttf'


## twitter API クラス
class TwitterApi:
    ## コンストラクタ
    ## 引数 検索キーワード, 1回のリクエストで取得する最大ツイート数
    def __init__(self, search_word, count):
        
        ## OAuth 認証
        self.twitter_api = OAuth1Session(
                consumer_api_key, 
                consumer_api_secret, 
                access_token, 
                access_token_secret
                ) 
        
        ## API 用 URL
        self.url = twitter_url
        
        self.params = {
                'q': search_word, 
                'count': count, 
                'result_type': 'recent', 
                'exclude': 'retweets', 
                'geocode': search_area
                }
        
        ## 取得ツイート数
        self.tweet_num = count
        
    ## 次のリクエストを実施
    def get_next_tweets(self):
        req = self.twitter_api.get(self.url, params=self.params)
        
        ## 正常に取得できている場合
        if req.status_code == 200:
            
            self.x_rate_limit_remaining = req.headers['X-Rate-Limit-Remaining']
            self.x_rate_limit_reset = req.headers['X-Rate-Limit-Reset']
            
            ## JSON データを辞書型として格納
            self.tweets = json.loads(req.text)
            
            self.tweet_num = len(self.tweets['statuses'])
            
            if self.tweet_num == 0:
                return True
            
            self.max_id = self.tweets['statuses'][0]['id']
            self.min_id = self.tweets['statuses'][-1]['id']
            
            next_max_id = self.min_id - 1
            
            self.params['max_id'] = next_max_id
            
            return True
        else:
            return False

    ## ツイートデータを取得
    def get_tweets_data(self):
        ## ツイートデータ格納データフレーム
        df_tweets = pd.DataFrame([])
        
        while self.tweet_num > 0:
            ret = self.get_next_tweets()
            
            ## ツイートデータがない場合、ループを抜ける
            if self.tweet_num == 0:
                break
            
            if ret:
                ## JSON の辞書型リストを DataFrame に変換
                df_temp = pd.io.json.json_normalize(self.tweets['statuses'])
                
                ## ツイートデータ格納データフレームに追加
                df_tweets = pd.concat([df_tweets, df_temp], axis=0, sort=False)
                
                print('アクセス可能回数 : ', self.x_rate_limit_remaining)
            else:
                ## エラー時はループを抜ける
                print('Error! : ', self.tweet_num)
                break
                
        ## ツイートデータをツイート日時 昇順に並び替えて返す
        return  df_tweets.sort_values('created_at').reset_index(drop=True)


## テキストマイニング　クラス
class TextMining:
    ## コンストラクタ
    def __init__(self, corpus):
        self.corpus = corpus    ## コーパス
        self.ignore_words = ignore_words    ## 除外する単語リスト
        
    ## コーパスから単語リストを抽出し、返す
    ## 引数に品詞リストを指定した場合、その品詞のみ抽出
    ## 返す単語は基本形（見出し語）
    def extract_words(self, word_class=None):
        t = Tokenizer()
        
        words = []
        
        ## 形態素解析と分かち書き
        for i in self.corpus:
            tokens = t.tokenize(i)
            
            for token in tokens:
                ## 品詞を抽出
                pos = token.part_of_speech.split(',')[0]
                
                ## 対象品詞リストがある場合、指定した品詞のみ抽出
                if word_class != None:
                    
                    ## 品詞リストから対象品詞のみ抽出
                    if pos in word_class:
                        
                        ## 除外する単語を除く
                        if token.base_form not in self.ignore_words:
                            words.append(token.base_form)
                            
                ## 対象品詞リストがない場合、全ての単語を抽出
                else:
                    words.append(token.base_form)
                    
        return words
    
    ## WordCloud
    def word_cloud(self, words, image_file, font_path=japanese_font):
        wordcloud = WordCloud(
                background_color='white', 
                font_path=font_path, 
                width=800, 
                height=400
                ).generate(words)
        
        ## 結果をファイルへ保存
        wordcloud.to_file(image_file)
        
        ## 結果を表示
        plt.imshow(wordcloud)
        plt.show()
        
        return True


## メイン
if __name__ == '__main__':
    ## twitter API クラス
    twitter_api = TwitterApi(search_word, count)
    
    ## ツイートデータを取得
    df_tweets = twitter_api.get_tweets_data()
    
    ## テキストマイニング　クラス
    text_mining = TextMining(df_tweets['full_text'])
    
    ## ツイートから対象品詞を抽出し、単語リストを得る
    words = text_mining.extract_words(word_class)
    
    ## WordCloud
    ## 単語リストをスペース区切りの変数に変換し渡す
    text_mining.word_cloud(' '.join(words), wordcloud_image_file)
