#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import time
import math
import scipy.stats
import pandas as pd

from container.simulator_container import SimulatorContainer
from container.logic_dict import LogicDict
from logic.data_wrangling.pre_processing.interface.i_pre_processing_input import IPreProcessingInput
from logic.analyze.sklearn.interface.i_sklearn_input import ISklearnInput
from logic.analyze.sklearn.enum.e_sklearn_type import ESklearnType 
from logic.analyze.light_gbm.interface.i_light_gbm_input import ILightGbmInput
from logic.analyze.light_gbm.enum.e_light_gbm_type import ELightGbmType 
from logic.analyze.xgboost.interface.i_xgboost_input import IXGBoostInput
from logic.analyze.xgboost.enum.e_xgboost_type import EXGBoostType 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# if (len(sys.argv) > 1) and (sys.argv[1] == "debug"):
#     import ptvsd
#     print("waiting...")
#     ptvsd.enable_attach("my_secret", address=('0.0.0.0', 8891))
#     ptvsd.wait_for_attach()

def execute():
    start = time.time()

    # データ事前処理
    iPreProcessingInput = IPreProcessingInput()
    iPreProcessingInput.input_train_path = '../../input/train.csv'
    iPreProcessingInput.input_test_path = '../../input/test.csv'
    iPreProcessingInput.output_x_train_path = 'data/output/X_train.pkl'
    iPreProcessingInput.output_y_train_path = 'data/output/Y_train.pkl'
    iPreProcessingInput.output_x_test_path = 'data/output/X_test.pkl'
    iPreProcessingInput.cv_value = 5

    # 新規にデータ事前処理をする場合はコメントアウト
    # X_train = pd.read_pickle("data/output/X_train.pkl")
    # Y_train = pd.read_pickle("data/output/Y_train.pkl")
    # iPreProcessingInput.X_train = X_train
    # iPreProcessingInput.Y_train = Y_train
    target_cols_list: list = get_list_all_pattern_count(
        ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize'],
        8)

    outputs: list = []

    for target_cols in target_cols_list:
        iPreProcessingInput.target_cols = target_cols

        # sklearn関連分析 ****************************************

        models = get_sklearn_models()

        execute_model_names = [
            'LogisticRegression',
            'LinearSVC',
            'SVC',
            'KNeighborsClassifier',
            'GaussianNB',
            'Perceptron',
            'DecisionTreeClassifier',
            'RandomForestClassifier'
        ]

        for model in models:
            if not model['model_name'] in execute_model_names:
                continue

            iPreProcessingInput.model = model['model_instance']
            iPreProcessingInput.grid_search_params = model['grid_search_params']
            iPreProcessingInput.random_search_params = model['random_search_params']
            iPreProcessingInput.sklearn_type = ESklearnType.CLASSIFICATION

            logic_dict: LogicDict = LogicDict(
                [
                    {
                        LogicDict.LOGIC_EXEC_KEY: 'PreProcessingLogic',
                        LogicDict.LOGIC_EXEC_INPUT_KEY: 'IPreProcessingInput',
                        LogicDict.LOGIC_EXEC_INPUT_INSTANCE: iPreProcessingInput,
                        LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ISklearnInput',
                    },
                    # {
                    #     LogicDict.LOGIC_EXEC_KEY: 'DataPatternLogic',
                    #     LogicDict.LOGIC_EXEC_INPUT_KEY: 'ISklearnInput',
                    #     LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ISklearnInput',
                    # },
                    {
                        LogicDict.LOGIC_EXEC_KEY: 'SklearnLogic',
                        LogicDict.LOGIC_EXEC_INPUT_KEY: 'ISklearnInput',
                        LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ISklearnOutput',
                    }
                ]
            )

            container = SimulatorContainer(logic_dict)
            container.execute()
            outputs.append(container.get_logic_output())


        # LightGBM分析  ****************************************
        iPreProcessingInput.light_gbm_type = ELightGbmType.CLASSIFICATION        
        logic_dict: LogicDict = LogicDict(
            [
                {
                    LogicDict.LOGIC_EXEC_KEY: 'PreProcessingLogic',
                    LogicDict.LOGIC_EXEC_INPUT_KEY: 'IPreProcessingInput',
                    LogicDict.LOGIC_EXEC_INPUT_INSTANCE: iPreProcessingInput,
                    LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ILightGbmInput',
                },
                # {
                #     LogicDict.LOGIC_EXEC_KEY: 'DataPatternLogic',
                #     LogicDict.LOGIC_EXEC_INPUT_KEY: 'ILightGbmInput',
                #     LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ILightGbmInput',
                # },
                {
                    LogicDict.LOGIC_EXEC_KEY: 'LightGbmLogic',
                    LogicDict.LOGIC_EXEC_INPUT_KEY: 'ILightGbmInput',
                    LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ILightGbmOutput',
                }
            ]
        )

        container = SimulatorContainer(logic_dict)
        container.execute()
        outputs.append(container.get_logic_output())


        # XGBoost分析  ****************************************
        iPreProcessingInput.xgboost_type = EXGBoostType.CLASSIFICATION        
        logic_dict: LogicDict = LogicDict(
            [
                {
                    LogicDict.LOGIC_EXEC_KEY: 'PreProcessingLogic',
                    LogicDict.LOGIC_EXEC_INPUT_KEY: 'IPreProcessingInput',
                    LogicDict.LOGIC_EXEC_INPUT_INSTANCE: iPreProcessingInput,
                    LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'IXGBoostInput',
                },
                # {
                #     LogicDict.LOGIC_EXEC_KEY: 'DataPatternLogic',
                #     LogicDict.LOGIC_EXEC_INPUT_KEY: 'IXGBoostInput',
                #     LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'IXGBoostInput',
                # },
                {
                    LogicDict.LOGIC_EXEC_KEY: 'XGBoostLogic',
                    LogicDict.LOGIC_EXEC_INPUT_KEY: 'IXGBoostInput',
                    LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'IXGBoostOutput',
                }
            ]
        )

        container = SimulatorContainer(logic_dict)
        container.execute()
        outputs.append(container.get_logic_output())

        break

    # 分析結果整形・出力  ****************************************

    output_model_infos: dict = {
        'model'         : [],
        'type'          : [],
        'score'         : [],
        'precision'     : [],
        'recall'        : [],
        'f1'            : [],
        'classification': [],
        'params'        : [],
        'target_cols'   : [],
    }
    for output in outputs:
        for result in output.results:
            output_model_infos['model'].append(output.model_name)
            output_model_infos['type'].append(result['Type'])
            output_model_infos['score'].append(result['Score'])

            if 'Precision' in result:
                output_model_infos['precision'].append(result['Precision'])
            else:
                output_model_infos['precision'].append(None)

            if 'Recall' in result:
                output_model_infos['recall'].append(result['Recall'])
            else:
                output_model_infos['recall'].append(None)

            if 'F1' in result:
                output_model_infos['f1'].append(result['F1'])
            else:
                output_model_infos['f1'].append(None)

            if 'Classification' in result:
                output_model_infos['classification'].append(result['Classification'])
            else:
                output_model_infos['classification'].append(None)

            output_model_infos['params'].append(result['Params'])
            if 'Target_cols' in result:
                output_model_infos['target_cols'].append(result['Target_cols'])           
            else:
                output_model_infos['target_cols'].append([])
            
    df = pd.DataFrame.from_dict(output_model_infos)
    df_s = df.sort_values('score', ascending=False)
    print(df_s)

    pd.to_pickle(df_s, 'data/output/df_score.pkl')
    # import pandas as pd
    # pd.read_pickle('../analysis_container/src/data/output/df_score.pkl')

    elapsed_time = math.floor(time.time() - start)
    models = get_sklearn_models()
    model_count = len(models) + 1
    target_cell_count = len(target_cols_list)
    total_execute_count = target_cell_count * model_count
    print ("target coll count  :{0}".format(target_cell_count) + "件")
    print ("model count        :{0}".format(model_count) + "件")
    print ("total execute count:{0}".format(total_execute_count) + "件")
    print ("1 elapsed_time     :{0}".format(math.floor(elapsed_time / total_execute_count)) + "[sec]")
    print ("total elapsed_time :{0}".format(elapsed_time) + "[sec]")

def get_sklearn_models() -> list:
    return [
        {
            'model_name': 'GradientBoostingRegressor',
            'model_instance': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5),
            'grid_search_params': {},
            'random_search_params': {}
        },
        {
            'model_name': 'Lasso',
            'model_instance': Lasso(alpha =0.0005, random_state=1),
            'grid_search_params': {},
            'random_search_params': {}
        },
        {
            'model_name': 'ElasticNet',
            'model_instance': ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3),
            'grid_search_params': {},
            'random_search_params': {}
        },
        {
            'model_name': 'KernelRidge',
            'model_instance': KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
            'grid_search_params': {},
            'random_search_params': {}
        },
        {
            # ロジスティック回帰(分類)
            'model_name': 'LogisticRegression',
            'model_instance': LogisticRegression(solver='liblinear'),
            'grid_search_params': {
                "C": [10 ** i for i in range(-5, 6)],
                "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
                "C": scipy.stats.uniform(0.00001, 1000),
                "random_state": scipy.stats.randint(0, 100)
            }
        },
        {
            # 線形SVC(分類)
            'model_name': 'LinearSVC',
            'model_instance': LinearSVC(dual=False),
            'grid_search_params': {
                "C": [10 ** i for i in range(-5, 6)],
                # "multi_class": ["ovr", "crammer_singer"],
                "class_weight": ["balanced"],
                # "random_state": [i for i in range(0, 101)]
                "random_state": [100]
            },
            'random_search_params': {
                "C": scipy.stats.uniform(0.00001, 1000),
                # "multi_class": ["ovr", "crammer_singer"],
                "class_weight": ["balanced"],
                # "random_state": scipy.stats.randint(0, 100)
                "random_state": [100]
            }
        },
        {
            # 非線形SVC(分類)
            'model_name': 'SVC',
            'model_instance': SVC(gamma="scale"),
            'grid_search_params': {
                "C": [10 ** i for i in range(-5, 6)],
                # "kernel": ["linear", "rbf", "sigmoid"],
                # "decision_function_shape": ["ovo", "ovr"],
                # "random_state": [i for i in range(0, 101)]
                "random_state": [100]
            },
            'random_search_params': {
                "C": scipy.stats.uniform(0.00001, 1000),
                # "kernel": ["linear", "rbf", "sigmoid"],
                # "decision_function_shape": ["ovo", "ovr"],
                # "random_state": scipy.stats.randint(0, 100)
                "random_state": [100]
            }
        },
        {
            # k近傍法
            'model_name': 'KNeighborsClassifier',
            'model_instance': KNeighborsClassifier(n_neighbors=3),
            'grid_search_params': {
                'weights': ['uniform', 'distance'],
                'n_neighbors': [3, 5, 10, 20, 30, 50]
            },
            'random_search_params': {
                'weights': ['uniform', 'distance'],
                'n_neighbors': [3, 5, 10, 20, 30, 50]
            }
        },
        {
            # Gaussian Naive Bayes
            'model_name': 'GaussianNB',
            'model_instance': GaussianNB(),
            'grid_search_params': {},
            'random_search_params': {}
        },
        {
            # Perceptron
            'model_name': 'Perceptron',
            'model_instance': Perceptron(),
            'grid_search_params': {
                # 'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                # 'n_iter': [5, 10, 15, 20, 50]
            },
            'random_search_params': {}
        },
        {
            # DecisionTreeClassifier
            'model_name': 'DecisionTreeClassifier',
            'model_instance': DecisionTreeClassifier(),
            'grid_search_params': {
                # "criterion": ["gini", "entropy"],
                # "splitter": ["best", "random"],
                # "max_depth": [i for i in range(1, 11)],
                # "min_samples_split": [i for i in range(2, 11)],
                # "min_samples_leaf": [i for i in range(1, 11)],
                # "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
            }
        },
        {
            # DecisionTreeRegressor
            'model_name': 'DecisionTreeRegressor',
            'model_instance': DecisionTreeRegressor(),
            'grid_search_params': {
                # "criterion": ["gini", "entropy"],
                # "splitter": ["best", "random"],
                # "max_depth": [i for i in range(1, 11)],
                # "min_samples_split": [i for i in range(2, 11)],
                # "min_samples_leaf": [i for i in range(1, 11)],
                # "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
            }
        },
        {
            # RandomForestClassifier
            'model_name': 'RandomForestClassifier',
            'model_instance': RandomForestClassifier(n_estimators=100),
            'grid_search_params': {
                # "criterion": ["gini", "entropy"],
                # "splitter": ["best", "random"],
                # "max_depth": [i for i in range(1, 11)],
                # "min_samples_split": [i for i in range(2, 11)],
                # "min_samples_leaf": [i for i in range(1, 11)],
                # "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
            }
        },
        {
            # RandomForestRegressor
            'model_name': 'RandomForestRegressor',
            'model_instance': RandomForestRegressor(n_estimators=100),
            'grid_search_params': {
                # "criterion": ["gini", "entropy"],
                # "splitter": ["best", "random"],
                # "max_depth": [i for i in range(1, 11)],
                # "min_samples_split": [i for i in range(2, 11)],
                # "min_samples_leaf": [i for i in range(1, 11)],
                # "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
            }
        },
    ]    

def get_list_all_pattern_count(patterns: list, limit_dimensionality_num: int = 2):
    all_pattern = [copy.deepcopy(patterns)]
    for i in range(len(patterns)):
        copy_patterns = copy.deepcopy(patterns)
        copy_patterns.pop(i)
        all_pattern.append(copy_patterns)

        if len(copy_patterns) > limit_dimensionality_num:
            all_pattern.extend  (get_list_all_pattern_count(copy_patterns, limit_dimensionality_num))
    
    return set(set(list(map(tuple,all_pattern))))

try:
    print('start')
    execute()

    # all_pattern: list = get_list_all_pattern_count(
    #     ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize'],
    #     7)
    # print(all_pattern)
    # print(len(all_pattern))
except Exception as e:
    print('error')
    print(e)
else:
    print('success')
finally:
    print('end')
