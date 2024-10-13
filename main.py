import pandas as pd
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

if __name__ == '__main__':

    # -------------------------------------------------------- Daten aus JSON in Korpus einlesen
    df = pd.read_json("data.json", lines=True) # load table from json into dataframe
    df = df[df.stars < 3] # remove 3, 4 and 5 star reviews
    corpus = df.text # load data from "text" column into list

    # -------------------------------------------------------- Preprocessing
    stemmer = SnowballStemmer("english")

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    # Tokenize and lemmatize
    def preprocess(text):
        result=[]
        for token in gensim.utils.simple_preprocess(text) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    processed_docs = []
    for doc in corpus:
        processed_docs.append(preprocess(doc))

    # -------------------------------------------------------- BoW
    dictionary = gensim.corpora.Dictionary(processed_docs)

    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # -------------------------------------------------------- LDA
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                    num_topics = 3, 
                                    id2word = dictionary,                                    
                                    passes = 50,
                                    workers = 2)

    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic ))
        print("\n")