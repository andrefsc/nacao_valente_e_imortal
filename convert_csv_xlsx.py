import pandas as pd

# Ler o arquivo CSV
caminho_csv =  '/Users/andre/Desktop/App_Closer/stats.csv'
dataframe = pd.read_csv(caminho_csv, delimiter=';')

print(dataframe)

# Converter para Excel
caminho_excel =  '/Users/andre/Desktop/App_Closer/stats.xlsx'
dataframe.to_excel(caminho_excel, index=False)

print("Arquivo Excel criado com sucesso!")