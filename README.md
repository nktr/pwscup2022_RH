# pwscup2022_RH
## ディレクトリ構成
- input
    - 入力ファイル置き場。知人ファイル・加工データのファイル・学習用データを置いておく
    - d0_main.csv: utiliy.csvと本戦でもらったorig_main_**.csvから作成size2000
    - d0.csv: 予備戦の漏洩データから作成size750
- output
    - 出力ファイル置き場。機械学習での攻撃リスト（attack_list_model.csv）・距離での攻撃リスト（attack_list_dist.csv）・距離＋機械学習での攻撃リスト（attack_list_merge_model_dist.csv）
- models_archive.sav
    - 学習済みモデルのアーカイブ

## 実行方法
make_attack_list.py input/(refファイル) input/(anonファイル)


