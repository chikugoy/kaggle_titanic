#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic

sys.path.append('/../../')


class LightGbmLogicClassification(AbstractLogic):

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        # LightGBMをインスタンス化
        gbm = lgb.LGBMClassifier(objective='binary')
        
        # X_trainとY_trainをtrainとvalidに分割
        train_x, valid_x, train_y, valid_y = train_test_split(
            self._input.X_train, self._input.Y_train, test_size=0.33, random_state=0)

        self._output.model_name = gbm.__class__.__name__
        self._logger.debug("モデル:{}".format(
            self._output.model_name) + " ***********************")

        # 試行するパラメータを羅列する
        params = {
            'max_depth' : [2, 3, 4, 5],
            'reg_alpha' : [0, 1, 10, 100],
            'reg_lambda': [0, 1, 10, 100],
        }

        grid_search = GridSearchCV(
            gbm,
            param_grid=params,  # 試行してほしいパラメータを渡す
            cv=3,               # 3分割交差検証でスコアを確認
        )

        grid_search.fit(self._input.X_train, self._input.Y_train)  # データを渡す
        pred_y = grid_search.predict(valid_x)

        self._logger.debug('サーチ方法:グリッドサーチ =======================')
        self._logger.debug("ベストスコア:{}".format(grid_search.best_score_))
        self._logger.debug("パラメーター:{}".format(grid_search.best_params_))
        results: list = []
        results.append({
            'Type'          : 'GridSearch',
            'Score'         : grid_search.best_score_,
            'Params'        : grid_search.best_params_,
            'Target_cols'   : self._input.X_train.columns,
            'Pred_y'        : pred_y
        })

        # X_trainとY_trainをtrainとvalidに分割
        train_x, valid_x, train_y, valid_y = train_test_split(
            self._input.X_train, self._input.Y_train, test_size=0.33, random_state=0)

        # trainとvalidを指定し学習
        gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                early_stopping_rounds=20,  # 20回連続でlossが下がらなかったら終了
                verbose=10  # 10round毎に、lossを表示
                )

        # valid_xについて推論
        # oofはout of fold
        oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)
        print('score', round(accuracy_score(valid_y, oof)*100, 2), '%')  # 正解率の表示

        # # testの予測
        # test_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)  # testの予測
        # sample_submission['Survived'] = test_pred  # sample_submissionのSurvived列をtest_predに置き換え
        # sample_submission.to_csv('train_test_split.csv', index=False)  # csvファイルの書き出し

        kf = KFold(n_splits=3)  # 3分割交差検証のためにインスタンス化

        # スコアとモデルを格納するリスト
        score_list          = []
        precision_list      = []
        recall_list         = []
        f1_list             = []
        classification_list = []
        models              = []

        print(self._input.X_train.columns)

        # 重要度
        print(pd.DataFrame({'特徴': self._input.X_train.columns,
                            'importance': gbm.feature_importances_}).sort_values('importance',
                                                                                 ascending=False))

        for fold_, (train_index, valid_index) in enumerate(kf.split(self._input.X_train, self._input.Y_train)):
            train_x = self._input.X_train.iloc[train_index]
            valid_x = self._input.X_train.iloc[valid_index]
            train_y = self._input.Y_train[train_index]
            valid_y = self._input.Y_train[valid_index]

            print(f'fold{fold_ + 1} start')

            # {'max_depth': 2, 'reg_alpha': 0, 'reg_lambda': 10}
            gbm = lgb.LGBMClassifier(objective='binary', max_depth=2, reg_alpha=0,
                                     reg_lambda=10, importance_type='gain')
            gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                    early_stopping_rounds=20,
                    # categorical_feature=['Sex', 'Embarked', 'Age'],   # ここもパターンしたい
                    verbose=-1)  # 学習の状況を表示しない

            oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

            score           = round(accuracy_score(valid_y, oof) * 100, 2) / 100
            precision       = round(precision_score(valid_y, oof) * 100, 2) / 100
            recall          = round(recall_score(valid_y, oof) * 100, 2) / 100
            f1              = round(f1_score(valid_y, oof) * 100, 2) / 100
            classification  = classification_report(valid_y, oof, output_dict=True)

            score_list.append(score)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            classification_list.append(classification)
            models.append(gbm)
            print(f'fold{fold_ + 1} end\n')

        print(score_list, '平均score', np.mean(score_list), "%")

        self._logger.debug('サーチ方法:デフォルトサーチ =======================')
        self._logger.debug("スコア:" + str(np.mean(score_list)))
        results.append({
            'Type'          : 'Default',
            'Score'         : np.mean(score_list) / 100,
            'Precision'     : np.mean(precision_list),
            'Recall'        : np.mean(recall_list),
            'F1'            : np.mean(f1_list),
            'Classification': classification_list,
            'Params'        : None,
            'Target_cols'   : self._input.X_train.columns,
            'Pred_y'        : None
        })

        self._output.results = results

        return True
