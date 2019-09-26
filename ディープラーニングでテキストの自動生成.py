# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:09:55 2019

@author: m_shibata
"""

import os
import re
import sys
import keras
import zipfile
import numpy as np
from keras import layers
from janome.tokenizer import Tokenizer

## 学習に使用するコーパス
## 銀河鉄道の夜, 風の又三郎, 注文の多い料理店
data_url = {
        'https://www.aozora.gr.jp/cards/000081/files/43737_ruby_19028.zip',
        'https://www.aozora.gr.jp/cards/000081/files/1943_ruby_29612.zip',
        'https://www.aozora.gr.jp/cards/000081/files/1927_ruby_17835.zip'
        }

## 訓練回数
epochs = 100

## 生成する単語数
generate_words = 100

## 確率的に次の単語を選択するためのパラメータ
temperature = 1.0

## テキストの始め4単語
first_sentences = '今日 の 天気 は'


## データをダウンロードし、読み込む
## テキストを改行コードで分割した文を保存する配列
text_array = []

for url in data_url:
    
    ## 青空文庫よりデータをダウンロード
    path = keras.utils.get_file(
            url.split('/')[-1],
            origin=url,
            cache_dir='./')
    
    ## ダウンロードしたデータを同じディレクトリに解凍（展開）
    zip = zipfile.ZipFile(path, 'r')
    file_name = zip.namelist()[0]
    zip.extractall(os.path.dirname(path))
    zip.close()
    
    ## ファイルを読み込む
    file = os.path.join(os.path.dirname(path), '', file_name)
    text = open(file, encoding='cp932').read()
    
    ## テキスト上部のタイトル・作者・記号の説明箇所を削除
    text = re.split(r'-+\n', text)[2]
    
    ## テキスト下部の「底本：」以降を削除
    text = re.split(r'底本：', text)[0]
    
    ## ルビを削除（正規表現での置換）
    text = re.sub(r'《.*?》|［＃.*?］|｜', '', text)
    
    ## テキスト前後の空白を削除
    text = text.strip()
    
    ## テキスト中の半角スペースを全角へ変換
    text = text.replace(' ', '　')
    
    ## テキストを改行コードで分割し、配列に追加
    text_array.extend(text.split('\n'))


## わかち書き
t = Tokenizer()

words = []

for i in text_array:
    for token in t.tokenize(i):
        words.append(token.surface)        ## 表層形
        #words.append(token.base_form)       ## 基本形・見出し語
        
text_wakachi = ' '.join(words)


## 文字列のベクトル化
maxlen = 4      ## 4単語のシーケンスを抽出
step = 1        ## 1単語おきに新しいシーケンスをサンプリング
sentences = []  ## 抽出されたシーケンスを保持
next_word = []  ## 目的値（次に来る単語）を保持

## 4単語のシーケンスを作成
for i in range(0, len(words) - maxlen, step):
    sentences.append(' '.join(words[i: i + maxlen]))
    next_word.append(words[i + maxlen])
print('シーケンス数 : ', len(sentences))

## コーパスのユニークな単語リスト
unique_words = sorted(list(set(words)))
print('ユニーク単語数 : ', len(unique_words))

## ユニークな単語リスト unique_words をインデックスとするディクショナリ
word_indices = dict((word, unique_words.index(word)) 
                                    for word in unique_words)

## 単語をベクトル化（One-Hot-Encoding）
print('ベクトル化...')
x = np.zeros((len(sentences), maxlen, len(unique_words)), dtype=np.bool)
y = np.zeros((len(sentences), len(unique_words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split(' ')):
        x[i, t, word_indices[word]] = 1
    y[i, word_indices[next_word[i]]] = 1


## 次の単語を予測する単層 LSTM モデル
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(unique_words))))
model.add(layers.Dense(len(unique_words), activation='softmax'))

## モデルのコンパイル設定
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


## モデルの予測に基づいて次の単語をサンプリングする関数
## スコアが最も高い単語を単純に選択するのではなく、ある程度のスコアからランダムに選択
## temperature が大きいと選択の幅が広がり、小さいと狭まる
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


## テキスト生成ループ
for epoch in range(1, epochs+1):
    print('学習回数 : ', epoch)
    ## 訓練毎にモデルを作成
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    generated_text = first_sentences
    sys.stdout.write(generated_text.replace(' ', ''))

    ## generate_words 個の単語を生成
    for i in range(generate_words):
        sampled = np.zeros((1, maxlen, len(unique_words)))
        for t, word in enumerate(generated_text.split(' ')):
            sampled[0, t, word_indices[word]] = 1.

        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_word = unique_words[next_index]

        generated_text += ' ' + next_word
        generated_text = ' '.join(generated_text.split(' ')[1:])
        
        sys.stdout.write(next_word)
        sys.stdout.flush()
        
    ## 改行
    print()
    