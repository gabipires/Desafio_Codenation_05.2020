import pandas as pd
import os
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import tree

df= pd.read_csv("C:/Users/GTP/Desktop/train.csv", sep=",", encoding="UTF8")
print(df.columns)

features = ["SG_UF_RESIDENCIA",
            "NU_IDADE",
            "TP_SEXO",
            "TP_COR_RACA",
            "TP_NACIONALIDADE",
            "TP_ST_CONCLUSAO",
            "TP_ANO_CONCLUIU",
            "TP_ESCOLA",
            "TP_ENSINO",
            "IN_TREINEIRO",
            "TP_DEPENDENCIA_ADM_ESC",
            "IN_BAIXA_VISAO",
            "IN_CEGUEIRA",
            "IN_SURDEZ",
            "IN_DISLEXIA",
            "IN_DISCALCULIA",
            "IN_SABATISTA",
            "IN_GESTANTE",
            "IN_IDOSO",
            "TP_PRESENCA_CN",
            "TP_PRESENCA_CH",
            "TP_PRESENCA_LC",
            "CO_PROVA_CN",
            "CO_PROVA_CH",
            "CO_PROVA_LC",
            "NU_NOTA_CN",
            "NU_NOTA_CH",
            "NU_NOTA_LC",
            "TP_LINGUA",
            "TP_STATUS_REDACAO",
            "NU_NOTA_COMP1",
            "NU_NOTA_COMP2",
            "NU_NOTA_COMP3",
            "NU_NOTA_COMP4",
            "NU_NOTA_COMP5",
            "NU_NOTA_REDACAO"]


alvo = "NU_NOTA_MT"

# Variáveis Categóricas
cd_cat = ["TP", "CO", "SG"]
cat_features = [i for i in features if i[:2] in cd_cat]
print(cat_features)


# Variáveis Numéricas
num_features = list(set(features) - set(cat_features))
#print(num_features)


# Removendo os valores nulos/NaN/null
df = df.dropna(how = 'all', subset = [alvo])

df[cat_features] = df[cat_features].fillna(-1).astype(str)
df[num_features] = df[num_features].fillna(-1)

df = df.reset_index(drop=True)

# Atribuindo valor as variáveis categóricas
onehot = preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
onehot.fit(df[cat_features])


df_onehot = pd.DataFrame(onehot.transform(df[cat_features]), columns=onehot.get_feature_names(cat_features))
print(df_onehot.head())


df_train = pd.concat([df[num_features], df_onehot], axis = 1, ignore_index= True)

regr = tree.DecisionTreeRegressor(max_depth= 12, min_samples_leaf=5)
print(regr.fit(df_train, df[alvo]))

features = df_train.columns.tolist()

model = pd.Series([num_features, cat_features, features, regr, onehot], index= ['num_features', 'cat_features', 'features', 'model', 'onehot'])

model.to_pickle('model.pkl')

df= pd.read_csv("C:/Users/GTP/Desktop/test.csv", sep=",", encoding="UTF8")

df[cat_features] = df[cat_features].fillna(-1).astype(str)
df[num_features] = df[num_features].fillna(-1)

df_onehot = pd.DataFrame(model ['onehot'].transform (df[ model ['cat_features']]), columns=model['onehot'].get_feature_names(model ['cat_features']))

df_full = pd.concat( [ df [ model[ 'num_features' ] ], df_onehot ], axis=1, ignore_index=True)

print(df_full)

predict = model['model'].predict(df_full[model['features']])

df_new = df[ ["NU_INSCRICAO"]].copy()
df_new['NU_NOTA_MT'] = predict

df_new.to_csv('answer.csv', index = False)




