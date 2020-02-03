参考
* https://qiita.com/zenonnp/items/9cbb2860505a32059d89
* https://qiita.com/taki_tflare/items/8850ac5ba8b504a171aa
* https://www.kaggle.com/currypurin/titanic-lightgbm
* https://qiita.com/morimori_japan/items/a62cad48ed4599815d3c

* http://data-analysis-stats.jp/2019/11/27/kaggle1%e4%bd%8d%e3%81%ae%e8%a7%a3%e6%9e%90%e6%89%8b%e6%b3%95%e3%80%80%e3%80%8c%e3%83%a1%e3%83%ab%e3%82%ab%e3%83%aa%e3%81%ab%e3%81%8a%e3%81%91%e3%82%8b%e5%80%a4%e6%ae%b5%e6%8e%a8%e5%ae%9a%e3%80%8d1/

* https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/

タイタニックデータ
* PassengerId：データにシーケンシャルでついている番号
* Survived：生存（0 = No, 1 = Yes）　訓練用データにのみ存在
* Pclass：チケットのクラス（1 = 1st, 2 = 2nd, 3 = 3rd）
* Name：名前
* Sex：性別
* Age：年齢
* SibSp：タイタニック号に乗っていた兄弟と配偶者の数
* Parch：タイタニック号に乗っていた両親と子どもの数
* Ticket：チケット番号
* Fare：旅客運賃
* Cabin：船室番号
* Embarked：乗船場（C = Cherbourg, Q = Queenstown, S = Southampton）



# 特微量分類

* 質的データ
    * 名義尺度
        * 他と区別し分類する為のもの
    * 順序尺度
        * 順序には意味があるが間隔に意味が無い物
* 量的データ
    * 間隔尺度
        * メモリが等間隔になっているもの
    * 比較尺度
        * 原点があり、間隔や比率に意味があるもの


* 質的データ  
Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
* 量的データ  
Continous: Age, Fare. Discrete: SibSp, Parch.


# データ事前処理

### - 特徴量にデータタイプが混合してるものがないか？  

数値と英数字の混合など。あれば基本数値に寄せる

### - エラーや誤植が含まれている特微量はないか？

### - 欠損値（空白やnullやNaN）が含まれている特微量はないか？

### - データセットでは数値タイプの特徴量の分布はどうなっているのか？  

pandasは下記で確認可能  

```
train_df.describe()
```

* 一意性と重複割合の確認
* 各列、または、複合列での比率の確認

### - 相関関係を調べる

* 相関性がないデータは削除を検討する

https://qiita.com/ShoheiKojima/items/9a94931d5f298cf9663c

個別相関
```
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

### - 連続的な数値的特徴量を順序尺度のカテゴリカル特徴量に変える

### - 統計量を確認  
https://qiita.com/ryo111/items/f62a43da6764f58823a5

### - 相関を確認

* 数値特徴量の相関
* 数値と順序尺度の特徴量の相関
* カテゴリカル特徴量の相関
* カテゴリカル特徴量と数値特徴量

### - 仮説を立てる

事前データの確認と分析対象に対するドメイン知識などから仮説を立てる
仮説立ては随時行う

# データラングリング

* 不要と思われる特微量の削除
* 既存のものから新しい特徴量を作成する
* カテゴリカル特徴量の変換
* 連続的数値の特徴量を補完する
* 既存の特徴量を組み合わせて新しい特徴量を作成する
* カテゴリカル特徴量を補完する
* カテゴリカル特徴量を数値に変換する
* クイック補完と数値特徴量の変換

# 分析

## scikit-learn

* LogisticRegression  
ロジスティック回帰  
* SVC(SVM Classification)  
SVMをクラス分類に用いる(SVMは回帰もできるので区別する。回帰はSVR: SVM Regression )
* RandomForestClassifier  
ランダムフォレスト
* KneighborsClassifier  
k近傍法
* GaussianNB  
ナイーブベイズのガウス分布版
* Perceptron  
パーセプトロン
* SGDClassifier  
確率的勾配降下(stochastic gradient descent)法を用いてSVMやロジスティック回帰を最適化する方法でクラス分類します。最適化については後で紹介する本に詳しく書かれていますので参照してください。
* DecisionTreeClassifier  
決定木
* LightGBM  
決定木アルゴリズムに基づいた勾配ブースティング（Gradient Boosting）の機械学習フレームワーク（LightGBMは米マイクロソフト社2016年にリリース）  
http://data-analysis-stats.jp/2019/11/13/lightgbm%E3%81%AE%E8%A7%A3%E8%AA%AC/


## 交差検証

https://qiita.com/taki_tflare/items/8850ac5ba8b504a171aa#%E4%BA%A4%E5%B7%AE%E6%A4%9C%E8%A8%BC%E3%82%92%E5%AE%9F%E6%96%BD%E3%81%97%E3%81%BE%E3%81%99

## ハイパーパラメーターの最適化

https://qiita.com/taki_tflare/items/8850ac5ba8b504a171aa#9-%E6%9C%80%E9%81%A9%E5%8C%96

