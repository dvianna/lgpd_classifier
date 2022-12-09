import pandas as pd
from preprocessing import PreprocessingData

file = open("training_data.txt", "w+")

dc = PreprocessingData(min_df=1, max_df=0.75, token_pattern=r'\b[a-zA-Z]{2,}\b', keep_url=False)

paths = ['/data/precedentes_obrigatorios/inteiroteor', '/data/Eleitoral1/inteiroteor', '/data/Eleitoral1/ementa',
         '/data/lgpd_2anos/lgpd_parquet', '/data/onu/injuriaracial_parquet', '/data/onu/racismo_parquet',
         '/data/racismo_parquet/inteiro_teor/']
for dir_path in paths:
    df = pd.read_parquet(dir_path, engine='pyarrow')
    df = df.loc[~df['html'].isnull()]
    documents = df['html'].tolist()
    preprocessed_corpus, list_of_ids = dc.prepare_text_for_topicmodel(documents, [])
    print(len(preprocessed_corpus))
    for doc in preprocessed_corpus:
        file.write(doc)

file.close()