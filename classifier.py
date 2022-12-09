import pandas as pd
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    AdaBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import PreprocessingData
import re


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def embed(embedding_model, documents, verbose):
    """ Embed a list of n documents/words into an n-dimensional
    matrix of embeddings

    Arguments:
        documents: A list of documents or words to be embedded
        verbose: Controls the verbosity of the process

    Returns:
        Document/words embeddings with shape (n, m) with `n` documents/words
        that each have an embeddings size of `m`
    """
    vector_shape = embedding_model.word_vec(list(embedding_model.vocab.keys())[0]).shape
    empty_vector = np.zeros(vector_shape[0])

    embeddings = []
    for doc in tqdm(documents, disable=not verbose, position=0, leave=True):
        doc_embedding = []

        # Extract word embeddings
        for word in doc.split(" "):
            try:
                word_embedding = embedding_model.word_vec(word)
                doc_embedding.append(word_embedding)
            except KeyError:
                doc_embedding.append(empty_vector)

        # Pool word embeddings
        doc_embedding = np.mean(doc_embedding, axis=0)
        embeddings.append(doc_embedding)

    embeddings = np.array(embeddings)
    return embeddings


def train_doc2vec(documents):
    min_count = 50
    hs = 1
    negative = 0
    epochs = 40
    doc2vec_args = {"vector_size": 300,
                    "min_count": min_count,
                    "window": 15,
                    "sample": 1e-5,
                    "negative": negative,
                    "hs": hs,
                    "epochs": epochs,
                    "dm": 0,
                    "dbow_words": 1}

    tokenized_doc = []
    for d in documents:
        tokenized_doc.append(word_tokenize(d, language='portuguese'))
    # Convert tokenized document into gensim formated tagged data
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]

    model = Doc2Vec(**doc2vec_args)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
    return model


def train_model_doc2vec():
    # inteiro-teor
    df = pd.read_parquet("/data/LGPD_jun/inteiroteor/", engine='pyarrow')
    df = df.loc[~df['html'].isnull()]
    documents = df['html'].tolist()
    dc = PreprocessingData(token_pattern=r"[a-zA-Z]\w*", keep_url=False)
    pp_documents = dc.clean_text(documents)
    doc2vec_model = train_doc2vec(pp_documents)
    doc2vec_model.save('doc2vec_model')


def tuning_hyperparameters(X, y, model_file_name):
    pipe = Pipeline([('classifier', GradientBoostingClassifier())])
    cross_val = StratifiedKFold(n_splits=5)
    search_space = [{'classifier': [GradientBoostingClassifier()],
                     'classifier__learning_rate': [0.01, 0.1],
                     'classifier__max_depth': [10, None],
                     "classifier__criterion": ["friedman_mse", "absolute_error"],
                     'classifier__n_estimators': [10, 100]
                     },
                    {'classifier': [LogisticRegression()],
                     'classifier__penalty': ['l1', 'l2'],
                     'classifier__solver': ['liblinear'],
                     'classifier__C': [0.01, 0.1, 1.0]},
                    {'classifier': [SVC()],
                     'classifier__kernel': ['linear'],
                     'classifier__C': [0.1, 1.0]},
                    {'classifier': [LGBMClassifier(random_state=0)],
                     'classifier__boosting_type': ['gbdt', 'dart', 'goss'],
                     'classifier__num_leaves': [20, 30],
                     'classifier__max_depth': [-1, 10, 20, 25],
                     'classifier__learning_rate': [0.001, 0.1],
                     'classifier__n_estimators': [10, 100]
                     }
                    ]

    clf = GridSearchCV(pipe, search_space, cv=cross_val, verbose=2, scoring='f1_macro', refit=True)
    clf = clf.fit(X, y)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    best_clf = clf.best_estimator_
    pickle.dump(best_clf, open(model_file_name, 'wb'))


def parameter_tuning(embedding_type='doc2vec', model_file_name='lgdp_classifier_doc2vec_multi', features='multi'):
    # load data
    doc_ids, documents, labels = load_data()
    # preprocessing documents
    # data cleaning
    dc = PreprocessingData(token_pattern=r"[a-zA-Z]\w*", keep_url=False)
    pp_documents = dc.clean_text(documents)
    # create feature vector
    embeddings = create_feature_vector(pp_documents, embedding_type=embedding_type, features=features)
    tuning_hyperparameters(embeddings, labels, model_file_name)


def create_feature_vector(documents, embedding_type='doc2vec', features='multi'):
    # get embeddings
    if embedding_type == 'doc2vec':
        model = train_doc2vec(documents)
        embeddings = model.docvecs.vectors_docs
    elif embedding_type == 'bert':
        emb_model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
        embeddings = emb_model.encode(documents, show_progress_bar=True)
    else:
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.75, stop_words=stopwords.words('portuguese'))
        embeddings = vectorizer.fit_transform(documents).toarray()

    if features == 'multi':
        extra_features = get_extra_features(documents)
        new_embedding = []
        for i, emb in enumerate(embeddings):
            new_embedding.append(np.concatenate((emb, extra_features[i])))
        embeddings = new_embedding
    return embeddings


def get_extra_features(documents):
    docs_extra_features = []
    for document in documents:
        extra_features = []
        # freq_law
        extra_features.append(len(re.findall("(?i)\w{0,}13.?709\w{0,}", document)))
        # freq_term
        extra_features.append(len(re.findall("(?i)\w{0,}LGPD\w{0,}", document)))
        # freq_phrase
        extra_features.append(len(re.findall("(?i)\w{0,}LEI GERAL DE PROTE[ÇC][ÃA]O DE DADO\w{0,}", document)))
        # document length (len_doc)
        len_doc = len(document)
        extra_features.append(len_doc)
        # freq_law/len_doc
        extra_features.append(len(re.findall("(?i)\w{0,}13.?709\w{0,}", document))/len_doc)
        # freq_term/len_doc
        extra_features.append(len(re.findall("(?i)\w{0,}LGPD\w{0,}", document))/len_doc)
        # freq_phrase/len_doc
        extra_features.append(len(re.findall("(?i)\w{0,}LEI GERAL DE PROTE[ÇC][ÃA]O DE DADO\w{0,}", document))/len_doc)
        docs_extra_features.append(extra_features)
    return docs_extra_features


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
    print(df.shape)
    # duplicateCheck = df.duplicated(subset=['Url:'], keep=False)
    # print(df[duplicateCheck])
    df.drop_duplicates(keep='last', subset=['Url:'], inplace=True)
    print(df.shape)
    exit()
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


def main():
    import torch
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parameter_tuning(embedding_type='doc2vec', model_file_name='lgdp_classifier_doc2vec_multi', features='multi')


if __name__ == '__main__':
    main()