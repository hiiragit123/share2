import colorsys


import pandas as pd
import numpy as np


from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pycld2 as cld2

from gensim.models import word2vec, KeyedVectors
from tqdm import tqdm

tqdm.pandas()

train = pd.read_csv('atmacup10_dataset/train.csv')
test = pd.read_csv('atmacup10_dataset/test.csv')
collection = pd.read_csv('atmacup10_dataset/object_collection.csv')
palette = pd.read_csv('atmacup10_dataset/palette.csv')
technique = pd.read_csv('atmacup10_dataset/technique.csv')
material = pd.read_csv('atmacup10_dataset/material.csv')

mat_col = pd.concat([material, collection], axis=0).reset_index(drop=True)
mat_tec = pd.concat([material, technique], axis=0).reset_index(drop=True)
col_tec = pd.concat([collection, technique], axis=0).reset_index(drop=True)
mat_col_tec = pd.concat([material, collection, technique], axis=0).reset_index(drop=True)


#複数テーブルの結合(作品の種類)
cross_object= pd.crosstab(collection['object_id'],collection['name'])
train = train.merge(cross_object, on='object_id', how='left')
test = test.merge(cross_object,on='object_id', how='left')


#色の種類数を求める
numr = []
numg = []
numb = []
for num in palette['color_r']:
    numr.append(num)
for num in palette['color_g']:
    numg.append(num)
for num in palette['color_b']:
    numb.append(num)

rgb = list(zip(numr,numg,numb))
hsvs = []
palette['color'] = np.zeros(len(palette))
for i,(x,y,z) in enumerate(rgb):
    hsv = colorsys.rgb_to_hsv(x,y,z)
    H = round(hsv[0]*360)
    hsvs.append(H//30)
    
palette['color'] = hsvs   
new_df = palette.groupby('object_id').nunique()
#複数テーブルの結合(色の種類数)
train = train.merge(new_df['color'], on='object_id', how='left')
test = test.merge(new_df['color'],on='object_id', how='left')

#word2vec
model_size = {
    "material": 20,
    "technique": 8,
    "collection": 3,
    "material_collection": 20,
    "material_technique": 20,
    "collection_technique": 10,
    "material_collection_technique": 25
}

n_iter = 100
w2v_dfs = []
for df, df_name in zip(
        [
            material, collection, technique,
            mat_col, mat_tec, col_tec, mat_col_tec
        ], [
            "material", "collection", "technique",
            "material_collection",
            "material_technique",
            "collection_technique",
            "material_collection_technique"
        ]):
    df_group = df.groupby("object_id")["name"].apply(list).reset_index()
    # Word2Vecの学習
    w2v_model = word2vec.Word2Vec(df_group["name"].values.tolist(),
                                  size=model_size[df_name],
                                  min_count=1,
                                  window=1,
                                  iter=n_iter)

    # 各文章ごとにそれぞれの単語をベクトル表現に直し、平均をとって文章ベクトルにする
    sentence_vectors = df_group["name"].progress_apply(
        lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
    sentence_vectors = np.vstack([x for x in sentence_vectors])
    sentence_vector_df = pd.DataFrame(sentence_vectors,
                                      columns=[f"{df_name}_w2v_{i}"
                                               for i in range(model_size[df_name])])
    sentence_vector_df.index = df_group["object_id"]
    w2v_dfs.append(sentence_vector_df)
for i in range(len(w2v_dfs)): 
    train = train.merge(w2v_dfs[i], on='object_id', how='left')
    test = test.merge(w2v_dfs[i], on='object_id', how='left')




def create_numeric_feature(input_df):
    use_columns = [
        'dating_period',
        'dating_year_early',
        'dating_year_late',
        'dollhouse', 
       'glass', 'jewellery',  'musical instruments', 'paintings',
       'paper', 'prints','color','material_w2v_0','material_w2v_1','material_w2v_2',
'material_w2v_3','material_w2v_4','material_w2v_5','material_w2v_6','material_w2v_7',
'material_w2v_8','material_w2v_9','material_w2v_10','material_w2v_11','material_w2v_12',
'material_w2v_13',
'material_w2v_14',
'material_w2v_15',
'material_w2v_16',
'material_w2v_17',
'material_w2v_18',
'material_w2v_19',
'collection_w2v_0',
'collection_w2v_1',
'collection_w2v_2',
'technique_w2v_0',
'technique_w2v_1',
'technique_w2v_2',
'technique_w2v_3',
'technique_w2v_4',
'technique_w2v_5',
'technique_w2v_6',
'technique_w2v_7',
'material_collection_w2v_0',
'material_collection_w2v_1',
'material_collection_w2v_2',
'material_collection_w2v_3',
'material_collection_w2v_4',
'material_collection_w2v_5',
'material_collection_w2v_6',
'material_collection_w2v_7',
'material_collection_w2v_8',
'material_collection_w2v_9',
'material_collection_w2v_10',
'material_collection_w2v_11',
'material_collection_w2v_12',
'material_collection_w2v_13',
'material_collection_w2v_14',
'material_collection_w2v_15',
'material_collection_w2v_16',
'material_collection_w2v_17',
'material_collection_w2v_18',
'material_collection_w2v_19',
'material_technique_w2v_0',
'material_technique_w2v_1',
'material_technique_w2v_2',
'material_technique_w2v_3',
'material_technique_w2v_4',
'material_technique_w2v_5',
'material_technique_w2v_6',
'material_technique_w2v_7',
'material_technique_w2v_8',
'material_technique_w2v_9',
'material_technique_w2v_10',
'material_technique_w2v_11',
'material_technique_w2v_12',
'material_technique_w2v_13',
'material_technique_w2v_14',
'material_technique_w2v_15',
'material_technique_w2v_16',
'material_technique_w2v_17',
'material_technique_w2v_18',
'material_technique_w2v_19',
'collection_technique_w2v_0',
'collection_technique_w2v_1',
'collection_technique_w2v_2',
'collection_technique_w2v_3',
'collection_technique_w2v_4',
'collection_technique_w2v_5',
'collection_technique_w2v_6',
'collection_technique_w2v_7',
'collection_technique_w2v_8',
'collection_technique_w2v_9',
'material_collection_technique_w2v_0',
'material_collection_technique_w2v_1',
'material_collection_technique_w2v_2',
'material_collection_technique_w2v_3',
'material_collection_technique_w2v_4',
'material_collection_technique_w2v_5',
'material_collection_technique_w2v_6',
'material_collection_technique_w2v_7',
'material_collection_technique_w2v_8',
'material_collection_technique_w2v_9',
'material_collection_technique_w2v_10',
'material_collection_technique_w2v_11',
'material_collection_technique_w2v_12',
'material_collection_technique_w2v_13',
'material_collection_technique_w2v_14',
'material_collection_technique_w2v_15',
'material_collection_technique_w2v_16',
'material_collection_technique_w2v_17',
'material_collection_technique_w2v_18',
'material_collection_technique_w2v_19',
'material_collection_technique_w2v_20',
'material_collection_technique_w2v_21',
'material_collection_technique_w2v_22',
'material_collection_technique_w2v_23',
'material_collection_technique_w2v_24'
    ]
    return input_df[use_columns].copy()


    
def create_string_length_feature(input_df): #タイトルなどの文字数
    out_df = pd.DataFrame()

    str_columns = [
        'title', 
        'long_title',
        'sub_title',
        'more_title',
        'description',
        
        # and more
    ]
    
    for c in str_columns:
        out_df[c] = input_df[c].str.len()

    return out_df.add_prefix('StringLength__')

def create_count_encoding_feature(input_df):
    use_columns = [
        'acquisition_method',
        'principal_maker',
        'principal_or_first_maker',
        'acquisition_credit_line'
        # and more
    ]

    out_df = pd.DataFrame()
    for column in use_columns:
        vc = train[column].value_counts()
        out_df[column] = input_df[column].map(vc)

    return out_df.add_prefix('CE_')

def label_encoder(input_df):
    use_columns = [
        'copyright_holder',
        'acquisition_method',
        'principal_maker',
        'principal_or_first_maker',
        'acquisition_credit_line'   
    ]
    
    out_df = pd.DataFrame()
    for column in use_columns:
        le = LabelEncoder()
        le.fit(input_df[column])
        out_df[column] = le.transform(input_df[column])
    
    return out_df.add_prefix('LE_')

def title_lan(input_df):
    use_columns = [
        'title'
    ]
    out_df = pd.DataFrame()
    for column in use_columns:
        features = input_df["title"].fillna("").map(lambda x: cld2.detect(x)[2][0][1])
        

        out_df[column] = features
        le = LabelEncoder()
        le.fit(out_df[column])
        out_df[column] = le.transform(out_df[column])
        for index,num in enumerate(out_df[column]):

     
            if num == 11:
                out_df.loc[index]= 1
            else:
                out_df.loc[index] = 0
    
       


    return out_df.add_prefix('TL_')
def hwdt(input):
    words = ['w','d','t']
    out_df = pd.DataFrame()
    for word in words:
        out_df[word] = np.zeros(len(input))
    
    for index,values in enumerate(input['sub_title']):

        if type(values) == float:
            continue
        for word in words:
            if word in values:

                out_df.loc[index][word] = 1
            else:

                out_df.loc[index][word] = 0
    return out_df

def hwdtsize(input):
    out_df = pd.DataFrame()
    for axis in ['h', 'w', 't', 'd']:
        column_name = f'size_{axis}'
        size_info = input['sub_title'].str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis)) # 正規表現を使ってサイズを抽出
        size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
        size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float) # dtypeがobjectになってるのでfloatに直す
        size_info[column_name] = size_info.apply(lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1) # 　単位をmmに統一する
        out_df[column_name] = size_info[column_name] 
    return out_df

#ここから新しいデータフレーム

def to_feature(input_df):
    """input_df を特徴量行列に変換した新しいデータフレームを返す.
    """

    processors = [
        create_numeric_feature,
        create_string_length_feature,
        create_count_encoding_feature,
        label_encoder,
        title_lan,
        hwdt,
        hwdtsize
    ]

    out_df = pd.DataFrame()

    for func in tqdm(processors, total=len(processors)):
        _df = func(input_df)

        # 長さが等しいことをチェック (ずれている場合, func の実装がおかしい)
        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)

    return out_df


    

train_feat_df = to_feature(train)
test_feat_df = to_feature(test)


train_feat_df.to_csv('new_dataset/train_feat.csv',index=False)
test_feat_df.to_csv('new_dataset/test_feat.csv',index=False)
