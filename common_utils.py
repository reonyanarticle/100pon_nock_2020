import csv
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import clear_output
from tensorflow.keras.callbacks import Callback

from typing import List


def make_csv(csv_path: str, words_list: List[List[str]], headers: List[str] = None) -> None:
    """
    文字列を要素とする二次元リストから、csvを作成する関数
    引数
    csv_path : str -> 作成したいcsvのパス
    words_list: List[List[str]] -> csvで保存したい文字列の二次元リスト
    headers : List[str] -> csvで保存する時につけるヘッダー名。デフォルトはNone
    """
    if os.path.exists(csv_path):
        print('{0}はもう存在しています'.format(csv_path))
        return None

    else:
        with open(csv_path, mode='w', encoding='utf-8') as f:
            writer = csv.writer(f)
            if headers is not None:
                writer.writerow(headers)

            writer.writerows(words_list)
        return None
    

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    vec1とvec2のコサイン類似度を求める関数
    引数
    vec1 : np.ndarray -> コサイン類似度を求めたい一つ目のベクトル
    vec2 : np.ndarray -> vc1とのコサイン類似度をもとめたいベクトル

    返り値
    cosine_similarity : float -> vec1とve2のコサイン類似度

    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def tokenize(word: str) -> str:
    """
    """
    word = word.replace('-', '')
    word = word.replace(',', '')
    word = word.replace('.', '')
    return word


class RealTimePlot(Callback):
    def __init__(self, name=False, absolute=False, log_scale=False, path=os.getcwd() ,save=False):
        super().__init__()
        self.name = name
        self.absolute = absolute
        self.log_scale = log_scale
        self.path = path
        self.save = save

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10,5))
        clear_output(wait=True)
        
        # fig全体の設定
        if self.name:
            fig.suptitle('{0}の学習'.format(self.name))
            
        
        # fig内のax1(サブグラフ)の設定
        if self.log_scale:
            ax1.set_yscale('log')
            
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        # fig内のax2(サブグラフ)の設定
        if self.absolute:
            ax2.set_ylim(0.6, 0.98)

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="val_accuracy")
        ax2.legend()
        plt.show();
        
    def on_train_end(self, logs={}):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10,5))
        clear_output(wait=True)
        
        # fig全体の設定
        if self.name:
            fig.suptitle('{0}の学習'.format(self.name))
            
        
        # fig内のax1(サブグラフ)の設定
        if self.log_scale:
            ax1.set_yscale('log')
    
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        # fig内のax2(サブグラフ)の設定
        if self.absolute:
            ax2.set_ylim(0.6, 0.98)

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="val_accuracy")
        ax2.legend()
        
        if self.save:
            output_fig_name = os.path.join(self.path, self.name + '.png')
            plt.savefig(output_fig_name)
        plt.show();