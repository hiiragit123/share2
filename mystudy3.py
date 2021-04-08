import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

train_feat_df = pd.read_csv('new_dataset/train_feat.csv')
test_feat_df = pd.read_csv('new_dataset/test_feat.csv')
sub = pd.read_csv('atmacup10_dataset/submission.csv')
delete_column = ['LE_copyright_holder']
train_feat_df.drop(delete_column, axis=1, inplace=True)
test_feat_df.drop(delete_column, axis=1, inplace=True)

train = pd.read_csv('atmacup10_dataset/train.csv')
y = train['likes'].values
y = np.log1p(y).round()
Y = train['likes'].values
Y = np.log1p(Y)
train['y'] = y

y_train = Y
X_train = train_feat_df
X_test = test_feat_df
categorical_features = ['LE_acquisition_method', 'LE_principal_maker',
       'LE_principal_or_first_maker', 'LE_acquisition_credit_line','TL_title','w','d','t', 'dollhouse', 
       'glass', 'jewellery',  'musical instruments', 'paintings',
       'paper', 'prints']

y_preds = []
y_preds = 0
models = []
oof_train = np.zeros((len(X_train),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train,y)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]
    c_train = Pool(X_tr,y_tr,cat_features=categorical_features)  
    c_test = Pool(X_val,y_val,cat_features=categorical_features)

    model = CatBoostRegressor(
            loss_function='RMSE'
        )
    model.fit(c_train)
    models.append(model)

    predict = model.predict(c_test)

