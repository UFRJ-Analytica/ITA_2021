# Importando Ferramentas Básicas
import pandas                  as pd
import matplotlib.pyplot       as plt
import numpy                   as np
import                            os
from   datetime            import datetime

# Importando Ferramentas de Limpeza
from sklearn.decomposition    import PCA
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import make_pipeline, Pipeline

# Importando Ferramentas de Modelo
from sklearn.svm              import SVR
from xgboost                  import XGBRegressor
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import GridSearchCV, RandomizedSearchCV
from sklearn.metrics          import accuracy_score, mean_absolute_error
from sklearn.linear_model     import LinearRegression, LogisticRegression, Lasso
from sklearn.base             import BaseEstimator


def prever(X_train, X_test, y_train, y_test, target_name):
    
    lista_scores = []
    lista_params = []
    lista_PCA = []
    lista_model = []
    
    components = [0.02,0.8,0.85,0.9,0.95,1]
    
    models = [
        LinearRegression(),
        #SVR(),
        #SVR(),
        Lasso()
        #XGBRegressor()
        ]
            
    for i, model in enumerate(models):

            for n in components:

                print(f"\n\nModelo: {model}\nComponent: {n}\n\n")

                pca = PCA(n_components = n)
                X_train_PCA = pca.fit_transform(X_train)
                X_test_PCA = pca.transform(X_test)

                clf = GridSearchCV(model, param_grid = params_grid[i],
                                   scoring = 'neg_mean_absolute_error', #destaque à métrica pedida
                                   n_jobs=2, refit=True, cv=5, verbose=5,
                                   pre_dispatch='2*n_jobs', error_score='raise', 
                                   return_train_score=True)

                clf.fit(X_train, y_train)

                pred_cv = clf.predict(X_test)
                score_cv = mean_absolute_error(y_test, pred_cv)
                print(f"Melhores parametros: {clf.best_params_}")
                print(f"\nScore Grid: {score_cv}")



                # clf_fit = model
                # params = clf_fit.set_params(**clf.best_params_)

                # clf_fit.fit(X_train, y_train)
                # y_pred = clf_fit.predict(X_test)
                # score = mean_absolute_error(y_test, y_pred)
                
                # print(f"Score: {score}")

                lista_model.append(model)
                lista_params.append(clf.best_params_)
                lista_scores.append(score_cv)
                lista_PCA.append(n)

    print("Exportando DataFrame de Scores\n")

    df_scores = pd.DataFrame(lista_scores)
    df_scores.insert(loc=0, column='PCA', value= pd.Series(lista_PCA))
    df_scores.insert(loc=0, column='Model', value= pd.Series(lista_model))
    df_scores.insert(loc=0, column='params', value= pd.Series(lista_params))
    df_scores.to_csv(f"./../Resultados/{target_name}_scores_"+"{}.csv".format(datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss")))
            
    return df_scores

def geral_resultados_submissao(test, clf_price, clf_trans):

    cent_price_cor = clf_price.predict(test.drop("id", axis=1))
    cent_trans_cor = clf_trans.predict(test.drop("id", axis=1))

    df_sub = pd.DataFrame({"cent_price_cor": cent_price_cor, "cent_trans_cor": cent_trans_cor})
    
    df_sub.to_csv("./../Submissoes/df_sub_{}.csv".format(datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss")), index=False)

    return df_sub



# Execucao do programa

# Importando os dados
train = pd.read_csv('./../Dados/train.csv')
test = pd.read_csv('./../Dados/test.csv')

dataframes = [train, test]

for df in dataframes:
    df['volume']  = df.x * df.y * df.z
    df['densidade'] = df.volume / df.n

X = train.drop(columns = ['cent_price_cor', 'cent_trans_cor'], axis = 1)

y_price = train.cent_price_cor
y_trans = train.cent_trans_cor

X_train, X_test, y_price_train, y_price_test = train_test_split(X,y_price,
                                                    test_size = 0.25,
                                                    random_state = 0)

X_train, X_test, y_trans_train, y_trans_test = train_test_split(X,y_trans,
                                                    test_size = 0.25,
                                                    random_state = 0)


params_grid = [

#Linear Regression
{'normalize': ['True', 'False'],
'fit_intercept': ['True', 'False']},
    
#Lasso
{'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}  

]

prever(X, X_test, y_trans, y_trans_test, "trans")

print("\nPrevisao para o Price concluida \n")

prever(X, X_test, y_price, y_price_test, "price")

clf_price = LinearRegression({'fit_intercept': 'True', 'normalize': 'True'})
clf_price.fit(X, y_price)

clf_trans = LinearRegression({'fit_intercept': 'True', 'normalize': 'True'})
clf_trans.fit(X, y_trans)

print(X)
print(test)

df_sub = geral_resultados_submissao(test, clf_price, clf_trans)

print(df_sub)

print("\nPrograma executado com sucesso \n")







# Coletanea de parametros para o GridSearch

# params_grid = [

# #Linear Regression
# {'normalize': ['True', 'False'],
# 'fit_intercept': ['True', 'False']},
    
# #SVR RBF
# {'kernel': ['rbf'],
# 'C':[0.1, 0.5, 1, 5, 10],
# 'degree': [3,8],
# 'coef0': [0.01,10,0.5],
# 'gamma': ('auto','scale'),
# 'epsilon': [0.1,0.2]},
    
# #SVR POLY
# {'kernel': ['poly'],
# 'C':[0.1, 0.5, 1, 5, 10],
# 'degree': [3,8],
# 'coef0': [0.01,10,0.5],
# 'gamma': ('auto','scale'),
# 'epsilon': [0.1,0.2]},
    
# #Lasso
# {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03],
# 'fit_alpha':[0.005, 0.02, 0.03, 0.05, 0.06]},  
    
# #XGBoost
# {'nthread':[4], #when use hyperthread, xgboost may become slower
# 'objective':['reg:linear'],
# 'learning_rate': [.03, 0.05, .07], #so called `eta` value
# 'max_depth': [5, 6, 7],
# 'min_child_weight': [4],
# 'silent': [1],
# 'subsample': [0.7],
# 'colsample_bytree': [0.7],
# 'n_estimators': [500]}
# ]