import pandas as pd

def datasetaq():

    #Consumption dataframe
    df_consumo = pd.read_csv(r"consumo_energia.csv")
    df_consumo['date'] = pd.to_datetime(df_consumo['din_instante'])
    df_consumo.set_index('date', inplace=True)
    df_consumo.sort_index(inplace=True)
    df_consumo.drop(['id_subsistema'],axis=1)

    #Generation dataframe
    df_geracao = pd.read_csv(r"geracao_energia.csv")
    df_geracao['date'] = pd.to_datetime(df_geracao['index'])
    df_geracao.set_index('date', inplace=True)
    df_geracao['month'] = df_geracao.index.month
    df_geracao['weekday'] = df_geracao.index.weekday
    df_geracao.sort_index(inplace=True)
    merged_df = pd.merge(df_geracao, df_consumo, on='date')
    return merged_df

