
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

### - 仮説を立てる

# 分析種類

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