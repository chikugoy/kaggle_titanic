
import sys
import copy
import time
import math
import pandas as pd

from container.simulator_container import SimulatorContainer
from container.logic_dict import LogicDict
from logic.analyze.sklearn.interface.i_sklearn_input import ISklearnInput
from logic.data_wrangling.pre_processing.interface.i_pre_processing_input import IPreProcessingInput
from logic.analyze.light_gbm.interface.i_light_gbm_input import ILightGbmInput

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
import scipy.stats

import lightgbm as lgb


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

            logic_dict: LogicDict = LogicDict(
                [
                    {
                        LogicDict.LOGIC_EXEC_KEY: 'PreProcessingLogic',
                        LogicDict.LOGIC_EXEC_INPUT_KEY: 'IPreProcessingInput',
                        LogicDict.LOGIC_EXEC_INPUT_INSTANCE: iPreProcessingInput,
                        LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ISklearnInput',
                    },
                    {
                        LogicDict.LOGIC_EXEC_KEY: 'DataPatternLogic',
                        LogicDict.LOGIC_EXEC_INPUT_KEY: 'ISklearnInput',
                        LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ISklearnInput',
                    },
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
        logic_dict: LogicDict = LogicDict(
            [
                {
                    LogicDict.LOGIC_EXEC_KEY: 'PreProcessingLogic',
                    LogicDict.LOGIC_EXEC_INPUT_KEY: 'IPreProcessingInput',
                    LogicDict.LOGIC_EXEC_INPUT_INSTANCE: iPreProcessingInput,
                    LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ILightGbmInput',
                },
                {
                    LogicDict.LOGIC_EXEC_KEY: 'DataPatternLogic',
                    LogicDict.LOGIC_EXEC_INPUT_KEY: 'ILightGbmInput',
                    LogicDict.LOGIC_EXEC_OUTPUT_KEY: 'ILightGbmInput',
                },
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

        break

    # 分析結果整形・出力  ****************************************

    output_model_infos: dict = {
        'model': [],
        'type': [],
        'score': [],
        'params': [],
        'target_cols': [],
    }
    for output in outputs:
        for result in output.results:
            output_model_infos['model'].append(output.model_name)
            output_model_infos['type'].append(result['Type'])
            output_model_infos['score'].append(result['Score'])
            output_model_infos['params'].append(result['Params'])
            if 'Target_cols' in result:
                output_model_infos['target_cols'].append(result['Target_cols'])           
            else:
                output_model_infos['target_cols'].append([])
            
    df = pd.DataFrame.from_dict(output_model_infos)
    df_s = df.sort_values('score', ascending=False)
    print(df_s)

    pd.to_pickle(df_s, 'data/output/df_score.pkl')

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
