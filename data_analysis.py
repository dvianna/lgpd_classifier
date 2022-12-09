import pandas as pd


file_path = './data/responses.csv'
df = pd.read_csv(file_path)

print(df.columns)
print(df.shape)

df.drop_duplicates(keep='last', subset=['Url:'], inplace=True)
print(df.shape)

"""
Pertinência do debate sobre a LGPD:
0 - não é decisão judicial
1 - não tem relação com a LGPD
2 - apenas menciona a LGPD
3 - debate incidental sobre a LGPD
4 - a LGPD é a questão central do caso
"""

str_relevance = ['0 - não é decisão judicial', '1 - não tem relação com a LGPD', '2 - apenas menciona a LGPD', '3 - debate incidental sobre a LGPD',
                 '4 - a LGPD é a questão central do caso']

for rel in str_relevance:
    df_label = df[df['Pertinência do debate sobre a LGPD:'] == rel]
    print('%s: %d' % (rel, df_label.shape[0]))

# df = df.loc[(df['Decisão merece artigo ou texto no blog:'] == 'Sim') & (df['Pertinência do debate sobre a LGPD:'] == '4 - a LGPD é a questão central do caso')]
# print(df[['Pertinência do debate sobre a LGPD:', 'Decisão merece artigo ou texto no blog:']])