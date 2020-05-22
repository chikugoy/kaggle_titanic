#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from scipy.stats import norm, skew
from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic
import sys

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

sys.path.append('/../../')


class PreProcessingLogic(AbstractLogic):
    """事前データ処理クラス

    Arguments:
        AbstractLogic {AbstractLogic} -- [description]
    """
    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        if (self._input.input_train_path is None or self._input.input_test_path is None or
            self._input.output_x_train_path is None or self._input.output_y_train_path is None or
                self._input.output_x_test_path is None):
            raise Exception('path Nothing')

        # TODO もっとキレイにしたいん
        try:
            if self._input.sklearn_type:
                self._output.sklearn_type = self._input.sklearn_type
                pass
        except Exception:
            pass

        try:
            if self._input.light_gbm_type:
                self._output.light_gbm_type = self._input.light_gbm_type
                pass
        except Exception:
            pass

        try:
            if self._input.xgboost_type:
                self._output.xgboost_type = self._input.xgboost_type
                pass
        except Exception:
            pass
        
        self._output.X_train = None
        if os.path.exists(self._input.output_x_train_path):
            self._output.X_train = pd.read_pickle(self._input.output_x_train_path)

        self._output.Y_train = None
        if os.path.exists(self._input.output_y_train_path):
            self._output.Y_train = pd.read_pickle(self._input.output_y_train_path)
        
        self._output.model = self._input.model if hasattr(self._input, 'model') else None
        self._output.cv_value = self._input.cv_value
        self._output.target_cols = self._input.target_cols
        self._output.grid_search_params = self._input.grid_search_params if hasattr(self._input, 'grid_search_params') else None
        self._output.random_search_params = self._input.random_search_params if hasattr(self._input, 'random_search_params') else None

        if (not self._output.X_train is None and not self._output.Y_train is None):
            return True

        train = pd.read_csv(self._input.input_train_path)
        test = pd.read_csv(self._input.input_test_path)

        self._logger.debug(train.columns.values)
        self._logger.debug("Before train.shape:{0} test.shape:{1}".format(train.shape, test.shape))

        #check the numbers of samples and features
        print("The train data size before dropping Id feature is : {} ".format(train.shape))
        print("The test data size before dropping Id feature is : {} ".format(test.shape))

        #Now drop the  'Id' colum since it's unnecessary for  the prediction process.
        train.drop("Id", axis = 1, inplace = True)
        test.drop("Id", axis = 1, inplace = True)

        #check again the data size after dropping the 'Id' variable
        print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
        print("The test data size after dropping Id feature is : {} ".format(test.shape))

        #Deleting outliers
        train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

        #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
        train["SalePrice"] = np.log1p(train["SalePrice"])

        ntrain = train.shape[0]
        y_train = train.SalePrice.values
        all_data = pd.concat((train, test)).reset_index(drop=True)
        all_data.drop(['SalePrice'], axis=1, inplace=True)
        print("all_data size is : {}".format(all_data.shape))

        all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

        all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
        all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
        all_data["Alley"] = all_data["Alley"].fillna("None")
        all_data["Fence"] = all_data["Fence"].fillna("None")
        all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

        #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))

        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            all_data[col] = all_data[col].fillna('None')

        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            all_data[col] = all_data[col].fillna(0)

        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            all_data[col] = all_data[col].fillna(0)
        
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            all_data[col] = all_data[col].fillna('None')

        all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
        all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
        all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
        all_data = all_data.drop(['Utilities'], axis=1)
        all_data["Functional"] = all_data["Functional"].fillna("Typ")
        all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
        all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
        all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
        all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
        all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
        all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

        #Check remaining missing values if any 
        all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
        missing_data.head()

        #MSSubClass=The building class
        all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

        #Changing OverallCond into a categorical variable
        all_data['OverallCond'] = all_data['OverallCond'].astype(str)

        #Year and month sold are transformed into categorical features.
        all_data['YrSold'] = all_data['YrSold'].astype(str)
        all_data['MoSold'] = all_data['MoSold'].astype(str)

        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
                'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
                'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
                'YrSold', 'MoSold')

        # process columns, apply LabelEncoder to categorical features
        for c in cols:
            lbl = LabelEncoder() 
            lbl.fit(list(all_data[c].values)) 
            all_data[c] = lbl.transform(list(all_data[c].values))

        # shape        
        print('Shape all_data: {}'.format(all_data.shape))

        # Adding total sqfootage feature 
        all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

        # Check the skew of all numerical features
        skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness.head(10)

        skewness = skewness[abs(skewness) > 0.75]
        print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            #all_data[feat] += 1
            all_data[feat] = boxcox1p(all_data[feat], lam)

        all_data = pd.get_dummies(all_data)
        print(all_data.shape)

        train = all_data[:ntrain]
        test = all_data[ntrain:]


        pd.to_pickle(train, self._input.output_x_train_path)
        pd.to_pickle(y_train, self._input.output_y_train_path)
        pd.to_pickle(test, self._input.output_x_test_path)

        self._output.X_train = train
        self._output.Y_train = y_train
        self._output.X_test = test

        return True
