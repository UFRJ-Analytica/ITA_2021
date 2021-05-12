# Importando Ferramentas Básicas
import pandas                  as pd
import matplotlib.pyplot       as plt
import numpy                   as np
import                            os
from   datetime            import datetime

# Importando Ferramentas de Limpeza
from sklearn.decomposition    import PCA
from sklearn.preprocessing    import StandardScaler

# Importando Ferramentas de Modelo
from sklearn.svm              import SVR
from xgboost                  import XGBRegressor
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import GridSearchCV
from sklearn.metrics          import mean_absolute_error
from sklearn.linear_model     import LinearRegression, Lasso

def importa_dados():
    # Importando os dados
    train = pd.read_csv('./../Dados/train.csv')
    test = pd.read_csv('./../Dados/test.csv')

    # Criando features
    dataframes = [train, test]

    # Criação de features
    for df in dataframes:
        df['volume']  = df.x * df.y * df.z
        df['densidade'] = df.volume / df.n
    
    return train, test

def prepara_fit(train, test):
    
    X = train.drop(columns = ['cent_price_cor', 'cent_trans_cor'], axis = 1)

    y_price = train["cent_price_cor"]
    y_trans = train["cent_trans_cor"]

    return X, y_price, y_trans


def prever(X, y, target_name, components = [20,21,22,23,24,25,26,27,28]):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size = 0.25,
                                                        random_state = 0)


    params_grid = [ #Linear Regression
                    {'normalize': ['True', 'False'],
                    'fit_intercept': ['True', 'False']}

                    #Lasso
                    #,{'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]} 
                    ]
    
    lista_scores = []
    lista_pca = []
    lista_params = []
    lista_models = []
    
    models = [
        LinearRegression(),
        #SVR(),
        #SVR(),
        #Lasso()
        #XGBRegressor()
        ]
     
    for n in components:
        
        pca = PCA(n_components = n)
        X_pca = pca.fit_transform(X)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
            
        for i, model in enumerate(models):

            print(f"\n\nModelo: {model}\nComponent: {n}\n\n" + str(X_pca.shape) + str(X_pca.shape))

            clf = GridSearchCV(model, param_grid = params_grid[i],
                               scoring = 'neg_mean_absolute_error', #destaque a  metrica pedida
                               n_jobs=-1, refit=True, cv=5, verbose=4,
                               pre_dispatch='2*n_jobs', error_score='raise', 
                               return_train_score=True)
            
            clf.fit(X_train_pca, y_train)

            pred_cv = clf.predict(X_test_pca)
            score_cv = mean_absolute_error(y_test, pred_cv)

            print(f"Melhores parametros: {clf.best_params_}")
            print(f"\nScore Grid: {score_cv}")
            
            lista_params.append(clf.best_params_)
            lista_models.append(model)
            lista_scores.append(round(score_cv,15))
            lista_pca.append(n)

    print("Exportando DataFrame de Scores\n")

    df_scores = pd.DataFrame()
    
    df_scores.insert(loc=0, column='PCA', value= pd.Series(lista_pca))
    df_scores.insert(loc=0, column='Scores', value= pd.Series(lista_scores))
    df_scores.insert(loc=0, column='Params', value= pd.Series(lista_params))
    df_scores.insert(loc=0, column='Model', value= pd.Series(lista_models))
    df_scores.to_csv(f"./../Resultados/{target_name}_scores_"+"{}.csv".format(datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss")))
            
    return df_scores


def gera_modelo(pca_price_n , pca_trans_n):
    
    pca_price = PCA(n_components = pca_price_n)
    pca_trans = PCA(n_components = pca_trans_n)

    train_price_pca = pca_price.fit_transform(X)
    train_trans_pca = pca_trans.fit_transform(X)

    clf_price = LinearRegression({'fit_intercept': 'True', 'normalize': 'True'})
    clf_price.fit(train_price_pca, y_price)

    clf_trans = LinearRegression({'fit_intercept': 'True', 'normalize': 'True'})
    clf_trans.fit(train_trans_pca, y_trans)

    test_price_pca = pca_price.fit_transform(test.drop("id", axis=1))
    test_trans_pca = pca_trans.fit_transform(test.drop("id", axis=1))
    
    return test_price_pca, test_trans_pca, clf_trans, clf_price
    

def geral_resultados_submissao(test_price_pca, test_trans_pca, clf_price, clf_trans):
    
    cent_price_cor = clf_price.predict(test_price_pca)
    cent_trans_cor = clf_trans.predict(test_trans_pca)


    df_sub = pd.DataFrame({"cent_price_cor": cent_price_cor, "cent_trans_cor": cent_trans_cor})
    
    df_sub.to_csv("./../Submissoes/df_sub_{}.csv".format(datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss")), index=False)

    return df_sub
    
#####################    Execução do programa     ####################### 

train, test = importa_dados()

X, y_price, y_trans = prepara_fit(train, test)

df_scores_price = prever(X, y_price, "price", components = [20,21,22,23,24,25,26,27,28])
df_scores_trans = prever(X, y_trans, "trans", components = [20,21,22,23,24,25,26,27,28])

pca_price_n = len(X.columns)
pca_trans_n = len(X.columns)

test_price_pca, test_trans_pca, clf_trans, clf_price = gera_modelo(pca_price_n, pca_trans_n)

df_sub = geral_resultados_submissao(test_price_pca, test_trans_pca, clf_price, clf_trans)

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