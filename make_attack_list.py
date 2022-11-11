import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.spatial.distance import chebyshev

n_ref = 80 # ref(知人)データのサイズ
archivefile = "models_archive.sav"
def make_attack_list(path_ref, path_anon):

    # データ取得
    df_ref = pd.read_csv(path_ref,names=["age","gender","race","income","education","veteran","noh","htn","dm","ihd","ckd","copd","ca"])
    df_anon = pd.read_csv(path_anon,names=["age","gender","race","income","education","veteran","noh","htn","dm","ihd","ckd","copd","ca"])
    df_ref = prepro(df_ref)
    df_anon = prepro(df_anon)
    if os.path.exists(archivefile):
        models = pickle.load(open(archivefile, "rb"))
    else:
        df_d0 = pd.read_csv("input/d0_main.csv")
        df_d0 = prepro(df_d0)

        # 学習用データとテスト用への分割
        train, test_orig = train_test_split(df_d0, test_size=0.2, random_state=0)
        train = train.reset_index(drop=True)
        test_orig = test_orig.reset_index(drop=True)
        test = test_orig.drop('covid', axis=1)
        features = train.drop(["covid"], axis=1)
        target = train["covid"]

        # モデルの作成と推定
        models = make_models(features, target)
        pickle.dump(models, open(archivefile, "wb"))
    pred = models_predict(models, df_ref)

    # 距離算出
    dists = calc_dist(df_ref, df_anon)

    # 機械学習の結果から攻撃リストを作成
    attack_ids_model = pred.sort_values(ascending=False).index[:int(n_ref/2)]
    attack_list_model = [1 if i in attack_ids_model else 0 for i in range(n_ref)]

    # 距離算出の結果から攻撃リストを作成
    attack_ids_dists = dists.sort_values().index[:int(n_ref/2)]
    attack_list_dists = [1 if i in attack_ids_dists else 0 for i in range(n_ref)]

    # 機械学習の予測結果とマージするためのあれこれ
    w = 0.5
    dists = 1-dists/max(dists)
    # 機械学習と距離のマージを行い攻撃リストを作成
    pred_merge_model_dists = w*pred + (1-w)*dists
    attack_ids_merge_model_dists = pred_merge_model_dists.sort_values(ascending=False).index[:int(n_ref/2)]
    attack_list_merge_model_dists = [1 if i in attack_ids_merge_model_dists else 0 for i in range(n_ref)]

    # csvファイルへの出力
    attack_list_model = pd.DataFrame(attack_list_model)
    attack_list_dists = pd.DataFrame(attack_list_model)
    attack_list_merge_model_dists = pd.DataFrame(attack_list_merge_model_dists)
    attack_list_model.to_csv('output/attack_list_model.csv', header=False, index=False)
    attack_list_dists.to_csv('output/attack_list_dist.csv', header=False, index=False)
    attack_list_merge_model_dists.to_csv('output/attack_list_merge_model_dist.csv', header=False, index=False)

def prepro(df):
    # [todo]前処理onehotencodeとか
    return df

def models_predict(models, df):
    
    test_pred = np.zeros((len(df), 5))

    for fold_, gbm in enumerate(models):
      test_pred[:, fold_] = gbm.predict(df) # dfを予測

    # 複数のモデルでの予測の平均値をとる
    pred = pd.Series(np.mean(test_pred, axis=1))
    
    return pred

def make_models(features, target):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # スコアとモデルを格納するリスト
    score_list = []
    models = []

    for fold_, (train_index, valid_index) in enumerate(kf.split(features, target)):    
        train_x = features.iloc[train_index]
        valid_x = features.iloc[valid_index]
        train_y = target[train_index]
        valid_y = target[valid_index]

        # lab.Datasetを使って、trainとvalidを作っておく
        lgb_train= lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(valid_x, valid_y)

        # パラメータを定義
        lgbm_params = {'objective': 'binary'}

        # lgb.trainで学習
        gbm = lgb.train(params=lgbm_params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_valid],
                        early_stopping_rounds=20,
                        verbose_eval=-1
                        )
        oof = (gbm.predict(valid_x) > 0.5).astype(int)
        score_list.append(round(accuracy_score(valid_y, oof)*100,2))
        models.append(gbm)  # 学習が終わったモデルをリストに入れておく
    return models

def calc_dist(df_ref, df_anon):
    # weights = [2, 2, 5, 10, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    weights = [2, 10, 5, 2, 5, 10, 5, 10, 10, 10, 10, 10, 10]
    dists = pd.Series([-1]*n_ref)
    for i, row_ref in df_ref.iterrows():
        min_d = 10e9
        for j, row_anon in df_anon.iterrows():
          d = dist_manhattan(row_ref, row_anon, weights)# 距離算出方法は選択
          if d < min_d:
              min_d = d
        dists[i] = min_d
    return dists

def dist_manhattan(s1, s2, weights=None):
    """
    Series s1, s2の距離（マンハッタン距離）を求める
    weightsによって各列の重みを指定する
    """
    if weights is None:
        weights = [1]*len(s1)
    d = abs(s1 - s2).dot(weights)
    return d

def dist_euc(s1, s2,weights=None):
  """
  Series s1, s2の距離（ユークリッド距離）を求める
  """
  if weights is None:
    weights = [1]*len(s1)
  s1 = s1.dot(weights)
  s2 = s2.dot(weights)
  return np.linalg.norm(s1-s2)

def dist_chebyshev(s1, s2,weights=None):
  """
  Series s1, s2の距離（チェビシェフ距離）を求める
  """
  if weights is None:
    weights = [1]*len(s1)
  s1 = s1.dot(weights)
  s2 = s2.dot(weights)
  return chebyshev(s1,s2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.argv[0], ' refファイル.csv anonファイル.csv ')
        exit(-1)
make_attack_list(sys.argv[1],sys.argv[2])