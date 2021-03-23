import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from sklearn.inspection import permutation_importance


train = pd.read_csv('atmacup10_dataset/train.csv')
y = train['likes'].values
y = np.log1p(y).round()
Y = train['likes'].values
Y = np.log1p(Y)
train['y'] = y


train_feat_df = pd.read_csv('new_dataset/train_feat.csv')
test_feat_df = pd.read_csv('new_dataset/test_feat.csv')
sub = pd.read_csv('atmacup10_dataset/submission.csv')
delete_column = ['LE_copyright_holder']
train_feat_df.drop(delete_column, axis=1, inplace=True)
test_feat_df.drop(delete_column, axis=1, inplace=True)


y_train = Y
X_train = train_feat_df
X_test = test_feat_df

y_preds = []
models = []
oof_train = np.zeros((len(X_train),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
params = params = {
"boosting_type":'gbdt',
"num_leaves":34,
"max_depth":-1,
"learning_rate":0.01,
"n_estimators":20000,
"objective":"regression",
"metric":"rmse", #mae
"force_col_wise":True,
"bin_construct_sample_cnt":2000,
"bagging_freq": 3,
"subsample":0.7,
"colsample_bytree":0.5,
"reg_alpha":.7,
"reg_lambda":.1,
"random_state":42,
"n_jobs":-1,
}
categorical_features = ['CE_acquisition_method',
       'CE_principal_maker', 'CE_principal_or_first_maker',
       'CE_acquisition_credit_line', 
       'LE_acquisition_method', 'LE_principal_maker',
       'LE_principal_or_first_maker', 'LE_acquisition_credit_line','TL_title','w','d','t', 'dollhouse', 
       'glass', 'jewellery',  'musical instruments', 'paintings',
       'paper', 'prints']
y_preds = 0


feature_importance_df = pd.DataFrame()
for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train,y)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]
    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)
    
    model = lgb.train(params, lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      verbose_eval = 100,
                      num_boost_round = 10000,
                      early_stopping_rounds=200)
    
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_preds += y_pred/5
    models.append(model)
    #ここから特徴量の重要度
    importance = pd.DataFrame(model.feature_importance(), index=X_train.columns, columns=['importance'])
    print(importance)
    importance_s = importance.sort_values('importance')
    feature_importance_df = pd.concat([feature_importance_df, importance_s], 
                                          axis=1, ignore_index=True)
print(feature_importance_df.sum(axis=1))
feature_importance_df['sum']= feature_importance_df.sum(axis=1)

#ここから各CVスコアの平均取得    
score = mean_squared_error(y,oof_train) ** .5
print('-' * 50)
print('FINISHED | Whole RMSLE: {:.4f}'.format(score))

pd.DataFrame(oof_train).to_csv('oof_train_kfold.csv', index=False)



y_preds = np.expm1(y_preds)
y_preds[y_preds<0] = 0
sub['likes'] =y_preds
sub.to_csv('first_try.csv', index=False) 


feature_importance_df.to_csv('FISUM.csv')