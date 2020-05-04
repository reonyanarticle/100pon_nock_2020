import csv
import os
import numpy as np

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
