#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import random as rnd
from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic
import sys

sys.path.append('/../../')


class PreProcessingLogic(AbstractLogic):

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        if (self._input.input_train_path is None or self._input.input_test_path is None or
            self._input.output_x_train_path is None or self._input.output_y_train_path is None or
                self._input.output_x_test_path is None):
            raise Exception('path Nothing')

        self._output.X_train = None
        if os.path.exists(self._input.output_x_train_path):
            self._output.X_train = pd.read_pickle(self._input.output_x_train_path)

        self._output.Y_train = None
        if os.path.exists(self._input.output_y_train_path):
            self._output.Y_train = pd.read_pickle(self._input.output_y_train_path)
        
        self._output.model = self._input.model
        self._output.cv_value = self._input.cv_value
        self._output.target_cols = self._input.target_cols
        self._output.grid_search_params = self._input.grid_search_params
        self._output.random_search_params = self._input.random_search_params

        if (not self._output.X_train is None and not self._output.Y_train is None):
            return True

        train_df = pd.read_csv(self._input.input_train_path)
        test_df = pd.read_csv(self._input.input_test_path)
        combine = [train_df, test_df]

        self._logger.debug(train_df.columns.values)
        self._logger.debug("Before train_df.shape:{0} test_df.shape:{1} combine[0].shape:{2} combine[1].shape:{3}".format(train_df.shape, test_df.shape,
              combine[0].shape, combine[1].shape))

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
            dataset.loc[(dataset['Age'] > 16) & (
                dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (
                dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (
                dataset['Age'] <= 64), 'Age'] = 3
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
        pd.to_pickle(train_df, 'data/output/train_df.pkl')
        pd.to_pickle(test_df, 'data/output/test_df.pkl')

        # データを読み込ませる準備
        # X_trainには応答変数（答えとなる特徴量）を除いた予測変数（応答変数を予測するために使う特徴量のこと）を入れる
        X_train = train_df.drop("Survived", axis=1)
        # Y_trainには応答変数のみを入れる
        Y_train = train_df["Survived"]
        # X_testにはテストデータを格納したデータフレームを入れ
        X_test = test_df.drop("PassengerId", axis=1).copy()
        X_train.shape, Y_train.shape, X_test.shape

        pd.to_pickle(X_train, self._input.output_x_train_path)
        pd.to_pickle(Y_train, self._input.output_y_train_path)
        pd.to_pickle(X_test, self._input.output_x_test_path)

        self._output.X_train = X_train
        self._output.Y_train = Y_train
        # self._output.X_test = X_test

        return True
