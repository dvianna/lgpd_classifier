from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from preprocessing import PreprocessingData
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


import torch
torch.manual_seed(2022)
torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            label = 1
        else:
            label = 0
        return label

    file_path = './data/responses.csv'
    df = pd.read_csv(file_path)
    df.drop_duplicates(keep='last', subset=['Url:'], inplace=True)
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

# create pandas dataframe
zipped = list(zip(doc_ids, pp_documents, labels))
df = pd.DataFrame(zipped, columns=['id', 'text', 'label'])
dataset_lgpd = Dataset.from_pandas(df)
dataset_lgpd = dataset_lgpd.class_encode_column("label")

dataset_lgpd = dataset_lgpd.train_test_split(stratify_by_column='label', test_size=0.20)

# Load SetFit model from Hub
model = SetFitModel.from_pretrained("rufimelo/Legal-BERTimbau-sts-base-ma-v2")
# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset_lgpd['train'],
    eval_dataset=dataset_lgpd['test'],
    loss_class=CosineSimilarityLoss,
    batch_size=4,
    num_iterations=20,  # The number of text pairs to generate
    column_mapping={"text": "text", "label": "label"}
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print(metrics)

y_pred = trainer.model(dataset_lgpd['test']['text'])
confusion_matrix(dataset_lgpd['test']['label'], y_pred)
print(classification_report(dataset_lgpd['test']['label'], y_pred))