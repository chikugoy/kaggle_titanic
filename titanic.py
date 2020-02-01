# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold

import lightgbm as lgb


def execute():
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')
    combine = [train_df, test_df]

    print(train_df.columns.values)
    print("Before", train_df.shape, test_df.shape,
          combine[0].shape, combine[1].shape)

    # 'Ticket', 'Cabin'は相関が無いため、特微量から削除
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    "After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

    # Titleの差し替え
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(
            ' ([A-Za-z]+)\.', expand=False)

    pd.crosstab(train_df['Title'], train_df['Sex'])

    # Titleを序数に変換
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    # Name削除
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    train_df.shape, test_df.shape

    # 文字列を含む特徴量を数値に変換
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map(
            {'female': 1, 'male': 0}).astype(int)

    train_df.head()

    # Pclass x Genderの組み合わせに基づいて推測されたAge値を格納
    # 2x3 Sex（0または1）とPclass（1,2,3）のAgeの中央値
    guess_ages = np.zeros((2, 3))
    guess_ages

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) &
                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    train_df.head()

    # 年齢を序数に置き換える
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    train_df.head()
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    train_df.head()

    # ParchとSibSpを組み合わせてFamilySizeとして新しい特徴量を作成
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Embarked欠損値を除外して最頻値で穴埋めをする
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    # Embarked特徴量を数値に変換
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map(
            {'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # 運賃の小数点第二位以下を四捨五入
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

    # FareBand特徴量を作成
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

    # FareBandに基づいてFare特徴量を序数に変換
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (
            dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (
            dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    print('before analyze data')
    print(train_df)

    # データを読み込ませる準備
    # X_trainには応答変数（答えとなる特徴量）を除いた予測変数（応答変数を予測するために使う特徴量のこと）を入れる
    X_train = train_df.drop("Survived", axis=1)
    # Y_trainには応答変数のみを入れる
    Y_train = train_df["Survived"]
    # X_testにはテストデータを格納したデータフレームを入れ
    X_test = test_df.drop("PassengerId", axis=1).copy()
    X_train.shape, Y_train.shape, X_test.shape

    models = []

    # ロジスティック回帰
    logreg = LogisticRegression(solver='liblinear')
    models.append(("LogisticRegression", logreg))
    # logreg.fit(X_train, Y_train)
    # # Y_pred = logreg.predict(X_test)
    # acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    # acc_log

    # coeff_df = pd.DataFrame(train_df.columns.delete(0))
    # coeff_df.columns = ['Feature']
    # coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

    # Support Vector Machines
    svc = SVC(gamma="scale")
    models.append(("SVC", svc))
    # svc.fit(X_train, Y_train)
    # # Y_pred = svc.predict(X_test)
    # acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    # acc_svc

    # k近傍法
    knn = KNeighborsClassifier(n_neighbors=3)
    models.append(("KNeighbors", knn))
    # knn.fit(X_train, Y_train)
    # # Y_pred = knn.predict(X_test)
    # acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    # acc_knn

    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    models.append(("GaussianNB", gaussian))
    # gaussian.fit(X_train, Y_train)
    # # Y_pred = gaussian.predict(X_test)
    # acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    # acc_gaussian

    # Perceptron
    perceptron = Perceptron()
    models.append(("Perceptron", perceptron))
    # perceptron.fit(X_train, Y_train)
    # # Y_pred = perceptron.predict(X_test)
    # acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    # acc_perceptron

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    models.append(("DecisionTreeClassifier", decision_tree))
    # decision_tree.fit(X_train, Y_train)
    # # Y_pred = decision_tree.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    # acc_decision_tree

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    models.append(("RandomForest", random_forest))
    # random_forest.fit(X_train, Y_train)
    # # Y_pred = random_forest.predict(X_test)
    # random_forest.score(X_train, Y_train)
    # acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    # acc_random_forest

    # models = pd.DataFrame({
    #     'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
    #             'Random Forest', 'Naive Bayes', 'Perceptron',
    #             'Decision Tree'],
    #     'Score': [acc_svc, acc_knn, acc_log,
    #             acc_random_forest, acc_gaussian, acc_perceptron,
    #             acc_decision_tree]})

    # print('models')
    # print(models.sort_values(by='Score', ascending=False))

    results = []
    names = []
    for name, model in models:
        # parameters = {'C' : [0.001, 0.01, 0.1, 1, 10, 100]}
        # gsc = GridSearchCV(model, parameters,cv=3)
        # gsc.fit(X_train, Y_train)

        # names.append(name)
        # results.append(gsc)
        result = cross_val_score(model, X_train, Y_train,  cv=3)
        names.append(name)
        results.append(result.mean())

    for i in range(len(names)):
        print(names[i], results[i])

    # sub = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])
    # sub['Survived'] = list(map(int, y_pred))
    # sub.to_csv('submission.csv', index=False)

    # =================================================================================
    # 勾配ブースティング決定木 (Gradient Boosting Decision Tree)

    # LightGBMの分類器をインスタンス化
    gbm = lgb.LGBMClassifier(objective='binary')

    # 試行するパラメータを羅列する
    params = {
        'max_depth': [2, 3, 4, 5],
        'reg_alpha': [0, 1, 10, 100],
        'reg_lambda': [0, 1, 10, 100],
    }

    grid_search = GridSearchCV(
                            gbm,  # 分類器を渡す
                            param_grid=params,  # 試行してほしいパラメータを渡す
                            cv=3,  # 3分割交差検証でスコアを確認
                            )

    grid_search.fit(X_train, Y_train)  # データを渡す

    print(grid_search.best_score_)  # ベストスコアを表示
    print(grid_search.best_params_)  # ベストスコアのパラメータを表示

    # X_trainとY_trainをtrainとvalidに分割
    train_x, valid_x, train_y, valid_y = train_test_split(
        X_train, Y_train, test_size=0.33, random_state=0)

    # trainとvalidを指定し学習
    gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],
            early_stopping_rounds=20,  # 20回連続でlossが下がらなかったら終了
            verbose=10  # 10round毎に、lossを表示
    ) ;

    # valid_xについて推論
    oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)  # oofはout of fold
    print('score', round(accuracy_score(valid_y, oof)*100,2), '%')  # 正解率の表示

    # # testの予測
    # test_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)  # testの予測
    # sample_submission['Survived'] = test_pred  # sample_submissionのSurvived列をtest_predに置き換え
    # sample_submission.to_csv('train_test_split.csv', index=False)  # csvファイルの書き出し

    kf = KFold(n_splits=3)  # 3分割交差検証のためにインスタンス化

    # スコアとモデルを格納するリスト
    score_list = []
    models = []

    print(X_train.columns)

    # 重要度
    print(pd.DataFrame({'特徴': X_train.columns,
        'importance':gbm.feature_importances_}).sort_values('importance',
        ascending=False))

    for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, Y_train)):
        train_x = X_train.iloc[train_index]
        valid_x = X_train.iloc[valid_index]
        train_y = Y_train[train_index]
        valid_y = Y_train[valid_index]
        
        print(f'fold{fold_ + 1} start')

        # {'max_depth': 2, 'reg_alpha': 0, 'reg_lambda': 10}
        gbm = lgb.LGBMClassifier(objective='binary', max_depth=2, reg_alpha=0,
                             reg_lambda=10, importance_type='gain')
        gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],
                    early_stopping_rounds=20,
                    categorical_feature=['Sex', 'Embarked', 'Age'],
                    verbose= -1) # 学習の状況を表示しない
        
        oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)
        score_list.append(round(accuracy_score(valid_y, oof)*100,2))
        models.append(gbm)  # 学習が終わったモデルをリストに入れておく
        print(f'fold{fold_ + 1} end\n' )

    print(score_list, '平均score', np.mean(score_list), "%")  

try:
    print('start')
    execute()
except Exception as e:
    print('error')
    print(e)
else:
    print('success')
finally:
    print('end')
