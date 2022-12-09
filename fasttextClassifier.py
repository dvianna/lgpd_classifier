import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import PreprocessingData


def load_data():
    def parse_url(text):
        text = text.strip('https://www.jusbrasil.com.br/jurisprudencia/')
        if 'inteiro-teor' in text:
            text = text.split('/inteiro-teor-')
            item = text[1]
        else:
            text = text.split('/')
            item = text[0]
        return int(item)

    lgpd_relevance = ['3 - debate incidental sobre a LGPD', '4 - a LGPD é a questão central do caso']
    lgpd_no_relevance = ['0 - não é decisão judicial', '1 - não tem relação com a LGPD', '2 - apenas menciona a LGPD']

    def add_label(lgpd):
        if lgpd in lgpd_relevance:
            label = '__label__positive'
        else:
            label = '__label__negative'
        return label

    file_path = './data/responses.csv'
    df = pd.read_csv(file_path)
    print(df.shape)
    # duplicateCheck = df.duplicated(subset=['Url:'], keep=False)
    # print(df[duplicateCheck])
    df.drop_duplicates(keep='last', subset=['Url:'], inplace=True)
    print(df.shape)
    df.dropna(subset=['Url:'], inplace=True)
    df.dropna(subset=['Pertinência do debate sobre a LGPD:'], inplace=True)
    df['idJurisprudencia'] = df['Url:'].apply(parse_url)
    df['labels'] = df['Pertinência do debate sobre a LGPD:'].apply(add_label)

    list_ids = df['idJurisprudencia'].tolist()
    print('Number of documents evaluated: ' + str(len(list_ids)))
    it_df = pd.read_parquet('/data/lgpd_2anos/lgpd_parquet', engine='pyarrow')
    it_df = it_df[it_df['idJurisprudencia'].isin(list_ids)]
    print('Number of documents (inteiro-teor): ' + str(len(list_ids)))

    df_merge = pd.merge(df, it_df, how='inner', on=['idJurisprudencia'])
    doc_ids = df_merge['idJurisprudencia'].tolist()
    documents = df_merge['html'].tolist()
    labels = df_merge['labels'].tolist()
    return doc_ids, documents, labels


# load data
doc_ids, documents, labels = load_data()

# preprocessing documents
# data cleaning
dc = PreprocessingData(token_pattern=r"[a-zA-Z]\w*", keep_url=False)
pp_documents = dc.clean_text(documents)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(pp_documents, labels, test_size=0.2, random_state=42)
# train data
data_tuples = list(zip(y_train, X_train))
df_train = pd.DataFrame(data_tuples, columns=['Label', 'Document'])
# test data
data_tuples = list(zip(X_test, y_test))
df_test = pd.DataFrame(data_tuples, columns=['Label', 'Document'])

df_train[['Label', 'Document']].to_csv('train.txt', index=False, sep=' ', header=None)

df_test[['Label', 'Document']].to_csv('test.txt', index=False, sep=' ', header=None)

# train classifier
model = fasttext.train_supervised('train.txt', wordNgrams=3, label='__label__')
model.test('train.txt', k=1)

