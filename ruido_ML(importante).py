#%% Bibliotecas e função geradora de ruído.
# Apenas coloque essa célula pra rodar no início
# do seu código. Ela já faz todo o trabalho de geração
# de ruído. Nas próximas células tem um exemplo de
# como o seu código deve ser.

import pandas as pd
import numpy as np

def ruido(df,fator_de_ruido):
    
    # Entradas:
        # 1 - df: banco de dados onde será acrescentado o ruído
        # 2 - fator_de_ruido: percentual máximo de ruído a ser inserido.
        
    # Zera o index de df, a função apresenta problemas
    # com index não sequenciais.
    df = df.reset_index(drop=True)
    
    # Nome das colunas.
    features = df.columns
    
    # Matriz de números aleatórios entre -0.5 e 0.5.
    aleatorio = np.random.random(df.shape) - 0.5
  
    # Transforma a matriz em DataFrame.
    aleatorio = pd.DataFrame(aleatorio)
    
    # Insere o nome das colunas no novo DataFrame.
    aleatorio.columns = features 

    # Média dos valores de df por coluna. É uma referência
    # para o ajuste de ordem de grandeza dos valores.
    media = df.mean(axis=0)

    # DataFrame vazio onde serão inseridos os resultados.
    nova = pd.DataFrame()
    
    # Aplicação da fórmula:
        # x = original + fr * media * aleatorio
    # Para cada coluna.
    for i in range(len(features)):
        nova[features[i]] = df.iloc[:,i] + fator_de_ruido*media[i]*aleatorio.iloc[:,i]
  
    # Retorna um DataFrame no mesmo formato que o de entrada,
    # porém com variação nos dados conforme o ruído inserido.
    return nova

#%% Importação do banco.

df = pd.read_csv("D:/Estudos Python/bancos de dados/milknew.csv")

# Do mesmo jeito que estamos acostumados.
x = df.drop(columns=['Grade','Colour'])
y = df['Grade']

#%% Separação de dados de treino e teste.

from sklearn.model_selection import train_test_split
x_treino,x_teste,y_treino,y_teste = train_test_split(x,y)

#%% Modelo random forest.


from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators = 10, criterion = 'gini', max_features = 'sqrt',random_state = 0)

# Treinamento.
modelo.fit(x_treino,y_treino)

# Predição dos dados de teste.
predicao = modelo.predict(x_teste)

# Acurácia.
from sklearn.metrics import accuracy_score
acuracia = accuracy_score(predicao,y_teste)

#%% Agora vem a novidade.

# Vou executar a função que criamos no início.
# Ela deve entrar com dois parâmetros:
# 1 - os dados que você quer inserir o ruído
# 2 - o percentual máximo do ruído.

x_ruido = ruido(x_teste,0.3)

# Queremos que modifique apenas os dados de teste,
# pois são sempre eles que estamos usando para a
# validação dos resultados.
# Compara x_treino com x_ruido, são muito parecidos,
# apenas com uma leve variação. É exatamente isso que
# queremos, vamos ver se o modelo aceita erros de medição.

# O modelo já está criado, TEMOS QUE USAR O MESMO.
# Queremos apenas investigar se esse modelo é robusto
# o suficiente pra acertar os resultados se tiver algum 
# erro de medição. Vamos testá-lo. 
predicao = modelo.predict(x_ruido)

# Por fim, vamos ver se ele acertou.
acuracia = accuracy_score(predicao,y_teste)

# Vá modificando o nível de ruído e olha o comportamento
# da acurácia. Me dê uma explicação da mudança.
