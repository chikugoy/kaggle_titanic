# 参考
# https://qiita.com/zenonnp/items/9cbb2860505a32059d89
# https://qiita.com/taki_tflare/items/8850ac5ba8b504a171aa
# https://www.kaggle.com/currypurin/titanic-lightgbm

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

* 特徴量にデータタイプが混合してるものがないか？  
数値と英数字の混合など

* エラーや誤植が含まれている特微量はないか？

* 欠損値（空白やnullやNaN）が含まれている特微量はないか？

* データセットでは数値タイプの特徴量の分布はどうなっているのか？

* 統計量を確認  
https://qiita.com/ryo111/items/f62a43da6764f58823a5



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