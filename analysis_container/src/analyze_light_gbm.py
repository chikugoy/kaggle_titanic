#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import time
import math
import pandas as pd

from container.simulator_container import SimulatorContainer
from container.logic_dict import LogicDict
from logic.data_wrangling.pre_processing.interface.i_pre_processing_input import IPreProcessingInput

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
    X_train = pd.read_pickle("data/output/X_train.pkl")
    Y_train = pd.read_pickle("data/output/Y_train.pkl")
    iPreProcessingInput.X_train = X_train
    iPreProcessingInput.Y_train = Y_train

    outputs: list = []
    iPreProcessingInput.target_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']

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
    print ("total elapsed_time :{0}".format(elapsed_time) + "[sec]")


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
