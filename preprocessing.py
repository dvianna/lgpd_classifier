import re
import unicodedata
import nltk
from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import spacy


# prepare data to run topic modeling solution
class PreprocessingData:
    def __init__(self, min_df=1, max_df=0.75, token_pattern=r"[a-zA-Z]\w*", keep_url=True, keep_title=True):
        """
        :param min_df:
        :param max_df:
        :param token_pattern: r"[a-zA-Z]\w*" - matches a word which contains at least one letter and zero or more digits,
        r'\b[a-zA-Z]{2,}\b' - matches only words containing letters, r"(?u)\b\w\w+\b" - keep letters and digits
        :param keep_url:
        :param keep_title:
        """

        self.min_df = min_df
        self.max_df = max_df
        self.token_pattern = token_pattern
        self.keep_url = keep_url
        self.keep_title = keep_title
        # list of stop words in portuguese
        nltk.download('stopwords')
        self.pt_stopwords = nltk.corpus.stopwords.words('portuguese')
        self.nlp = spacy.load("pt_core_news_lg")

        """
        The options below are specific for the project Pesquisa Pronta (dados do STJ)
        """
        # remove states names from text
        self.pt_stopwords.extend(["amazonas", "roraima", "amapá", "pará", "tocatins", "rondônia", "acre", "maranhão",
                                  "piauí", "ceará", "rio grande do norte", "pernambuco", "paraíba", "sergipe",
                                  "alagoas", "bahia", "mato grosso", "mato grosso do sul", "goiás", "são paulo",
                                  "rio de janeiro", "espírito santo", "minas gerais", "paraná", "rio grande do sul",
                                  "santa catarina", "rio grande sul", "rio grande", "sao paulo", "rio", "sp", "rondonia",
                                  "maranhao", "piaui", "ceara", "paraiba", "goias", "espirito santo", "parana", "catarina"])
        # remove a few names
        self.pt_stopwords.extend(["celsodemello", "marcoaurélio", "carlosbritto", "ricardolewandowski", "joaquimbarbosa",
                                  "ellengracie", "gilmarmendes", "carlosbritto", "cezarpeluso", "nelsonjobim",
                                  "ricardolewandowski", "octaviogallotti", "sepúlvedapertence", "moreiraalves",
                                  "néridasilveira", "carlosvelloso", "marcoaurelio", "sydneysanches", "sepulvedapertence",
                                  "ilmargalvão", "paulobrossard", "celsodemello", "moreiraalves", "octaviogallotti",
                                  "mauríciocorrêa", "fux", "barroso", "roberto barroso", "rosa weber", "weber",
                                  "robertobarroso", "rosaweber", "alexandredemoraes", "alexandre de moraes", "moraes",
                                  "joaquim barbosa", "barbosa", "celso de mello", "mello", "ricardo lewandowski",
                                  "lewandowski", "gilmar mendes", "mendes", "luizfux", "luiz fux", "edson fachin",
                                  "edsonfachin", "fachin", "celso", "joaquim", "gilmar", "rosa", "roberto",
                                  "diastoffoli", "toffoli", "alexandre"])

    def remove_names(self, text):
        """
        Using NER (spacy) to identify person removing them from the document html
        :param text: html (inteiro-teor)
        :return: html without person
        """
        try:
            doc = self.nlp(text)
            # creating the filter list for tokens that are identified as person
            fil = [ent.text for ent in doc.ents if ent.label_ in ["PER"]]
            for item in fil:
                text = text.replace(item, '')
        except:
            print('Text is too long')
            pass
        return text

    def mask_names(self, text):
        """
        Using NER (spacy) to identify person replacing them with masks (NOME)
        :param text: html (inteiro-teor)
        :return: html with masks for person (NOME)
        """
        try:
            doc = self.nlp(text)
            # creating the filter list for tokens that are identified as person
            fil_per = [ent.text for ent in doc.ents if ent.label_ in ["PER"]]
            for item in fil_per:
                text = text.replace(item, 'NOME')
        except:
            print('Text is too long')
            pass
        return text

    def remove_partes(self, html):
        """
        Removing "partes do processo" (designed to work with documents from STJ)
        :param html: html (inteiro-teor)
        :return: html withour "partes do processo"
        """

        # regex to eliminate "partes do processo" from STJ documents
        list_partes = [r"RECORRENTE.+?(?=\</p>)", r"RECORRIDO.+?(?=\</p>)", r"AGRAVANTE.+?(?=\</p>)",
                            r"AGRAVADO.+?(?=\</p>)", r"EMBARGANTE.+?(?=\</p>)", r"EMBARGADO.+?(?=\</p>)",
                            r"SUSCITANTE.+?(?=\</p>)", r"SUSCITADO.+?(?=\</p>)", r"RELATOR.+?(?=\</p>)",
                            r"ADVOGADO.+?(?=\</p>)", r"RECORRENTES.+?(?=\</p>)", r"RECORRIDOS.+?(?=\</p>)",
                            r"AGRAVANTES.+?(?=\</p>)", r"AGRAVADOS.+?(?=\</p>)", r"EMBARGANTES.+?(?=\</p>)",
                            r"EMBARGADOS.+?(?=\</p>)", r"SUSCITANTES.+?(?=\</p>)", r"SUSCITADOS.+?(?=\</p>)",
                            r"RELATORES.+?(?=\</p>)", r"ADVOGADOS.+?(?=\</p>)", r"RECORRENTE.+?(?=\</tr>)",
                            r"RECORRIDO.+?(?=\</tr>)", r"AGRAVANTE.+?(?=\</tr>)", r"AGRAVADO.+?(?=\</tr>)",
                            r"EMBARGANTE.+?(?=\</tr>)", r"EMBARGADO.+?(?=\</tr>)", r"SUSCITANTE.+?(?=\</tr>)",
                            r"SUSCITADO.+?(?=\</tr>)", r"RELATOR.+?(?=\</tr>)", r"ADVOGADO.+?(?=\</tr>)",
                            r"RECORRENTES.+?(?=\</tr>)", r"RECORRIDOS.+?(?=\</tr>)", r"AGRAVANTES.+?(?=\</tr>)",
                            r"AGRAVADOS.+?(?=\</tr>)", r"EMBARGANTES.+?(?=\</tr>)", r"EMBARGADOS.+?(?=\</tr>)",
                            r"SUSCITANTES.+?(?=\</tr>)", r"SUSCITADOS.+?(?=\</tr>)", r"RELATORES.+?(?=\</tr>)",
                            r"ADVOGADOS.+?(?=\</tr>)", r"RELATORA.+?(?=\</p>)", r"RELATORA.+?(?=\</p>)"]

        parsed_html = html
        for term in list_partes:
            parsed_html = re.sub(term, '', parsed_html)
        return parsed_html

    def remove_punctuation(self, text):
        """
        Remove punctuations
        :parameters:
            text: sentence (string)
        :return: preprocessed documents without punctuations
        """
        punctuation = r'[/.!$%^&#*+\'\"()-.,:;<=>?@[\]{}|]'
        return re.sub(punctuation, ' ', text)

    def strip_accents_from_word(self, text):
        """
        Strip accents from input String.
        :parameters:
            text: word (string)
        :return: preprocessed word without accent
         """
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return str(text)

    def strip_accents(self, text):
        """
        Strip accents from text.
        :parameters:
            text: sentence (string)
        :return: preprocessed documents without accent
         """
        return " ".join([self.strip_accents_from_word(word) for word in text.split()])

    def remove_URL(self, text):
        """
        Remove URLs from text
        :return: text without URLs
        """
        return re.sub(r"<.*?>", " ", text)

    def extract_text_from_html(self, html):
        """
        Extract text from html
        :return: text (clean html)
        """
        return BeautifulSoup(html).get_text()

    def parse_url(self, html, add_title=False):
        """
        Extract parts of URLs
        :return: parsed URLs
        """
        out = ''
        # example: http://www.jusbrasil.com.br/topicos/10619340/artigo-157-do-decreto-lei-n-2848-de-07-de-dezembro-de-1940
        link = html.get("href")
        # example: Artigo 157 do Decreto Lei nº 2.848 de 07 de Dezembro de 1940
        title_data = html.get("title")
        # example: http://www.jusbrasil.com.br/topicos/10619340/artigo-157-do-decreto-lei-n-2848-de-07-de-dezembro-de-1940
        parsed_url = urlparse(link)
        # example: www.jusbrasil.com.br
        netloc = parsed_url.netloc
        try:
            if 'jusbrasil.com.br' in netloc:
                try:
                    # example: /topicos/10619340/artigo-157-do-decreto-lei-n-2848-de-07-de-dezembro-de-1940
                    path = parsed_url.path
                    # example:
                    items = path.split('/')
                    # example: topicos_10619340
                    item_1 = items[1] + '_' + items[2]
                    # example: artigo-157-do-decreto-lei-n-2848-de-07-de-dezembro-de-1940
                    item_2 = items[3]
                    if add_title and title_data:
                        out = out + ' ' + title_data
                    if item_1:
                        out = out + ' ' + item_1
                    if item_2:
                        item_2 = item_2.replace('-', '_')
                        #item_2 = item_2.replace('-', ' ')
                        out = out + ' ' + item_2
                except:
                    print('Link came in an unexpected format!')
                    print(html)
                    pass
        except:
            print('Link came in an unexpected format!')
            print(html)
            pass
        return out

    def clean_url(self, text, add_title=False):
        """
        Parse text extracting URLs. We can either keep titles or not.
        :return: raw text plus parsed URLs (and maybe URLs' titles)
        """
        stripped_input = text
        soup = BeautifulSoup(text, "html.parser")
        for link in soup.findAll('a'):
            parts = self.parse_url(link, add_title)
            stripped_input = soup.get_text(separator=" ")
            if parts:
                stripped_input = stripped_input + ' ' + parts
        return stripped_input

    def clean_text(self, documents):
        """
        Clean (preprocess) text data: parse URL, remove punctuation, convert to lowercase, remove stopword, and
        strip accents
        :param documents: list of documents
        :return: preprocessed_text: list of preprocessed documents
        """
        # preprocess jusbrasil URLs
        preprocessed_text = []
        for text in documents:
            if self.keep_url:
                preprocessed_text.append(self.clean_url(text, self.keep_title))
            else:
                preprocessed_text.append(self.extract_text_from_html(text))
                #preprocessed_text.append(self.remove_URL(text))

        # agravodeinstrumentoai70068698984rs or lei_de_criacao_do_pis_lei_complementar_7_70
        preprocessed_text = [self.remove_punctuation(doc) for doc in preprocessed_text]
        # convert to lowercase
        preprocessed_text = [doc.lower() for doc in preprocessed_text]
        # remove stopwords
        preprocessed_text = [' '.join([w for w in doc.split() if len(w) > 1 and w not in self.pt_stopwords])
                             for doc in preprocessed_text]

        # strip accents
        # preprocessed_text = [self.strip_accents(doc) for doc in preprocessed_text]
        return preprocessed_text

    def prepare_text_for_topicmodel(self, documents, list_doc_ids):
        """
        Prepare data for BERTopic topic discovery
        :param documents: list of documents
        :param list_doc_ids: list of documents' ids
        :return: preprocessed_docs: list of preprocessed documents
                 ids_docs_removed: list of ids to be removed
        """

        # the couple of lines below were attempts to improve the quality of the keywords by masking/deleting names
        #preprocessed_docs_tmp = [self.mask_names(doc) for doc in documents]
        #preprocessed_docs_tmp = [self.remove_partes(doc) for doc in documents]

        preprocessed_docs_tmp = documents

        # clean the data: parser URL, lowercase, remove punctuation, and remove stopwords
        preprocessed_docs_tmp = self.clean_text(preprocessed_docs_tmp)

        # stop_word = None. Stopwords were removed in clean_text()
        vectorizer = CountVectorizer(stop_words=None,
                                     tokenizer=None,
                                     min_df=self.min_df,
                                     max_df=self.max_df,
                                     ngram_range=(1, 2),
                                     token_pattern=self.token_pattern,
                                     lowercase=True)

        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary = set(vectorizer.get_feature_names())

        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w.lower() in vocabulary])
                                 for doc in preprocessed_docs_tmp]

        list_ids = []
        preprocessed_docs, unpreprocessed_docs = [], []
        count_deleted = 0
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                list_ids.append(list_doc_ids[i])
                unpreprocessed_docs.append(documents[i])
            else:
                count_deleted += 1
        print("Number of documents being removed because of their size: %d" % count_deleted)
        return preprocessed_docs, list_ids

    def prepare_text_for_bertopic(self, documents):
        """
        Prepare data for BERTopic topic discovery
        :param documents: list of documents
        :param list_doc_ids: list of documents' ids
        :return: preprocessed_docs: list of preprocessed documents
                 ids_docs_removed: list of ids to be removed
        """
        # clean the data: parser URL, lowercase, remove punctuation, and remove stopwords
        preprocessed_text = []
        for text in documents:
            if self.keep_url:
                preprocessed_text.append(self.clean_url(text, self.keep_title))
            else:
                preprocessed_text.append(self.remove_URL(text))

        return preprocessed_text, []
