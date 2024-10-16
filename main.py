import pandas as pd
import re
import nltk
import string
import gensim

import logging

if __name__ == "__main__":
    PYTHONHASHSEED=0 # zwecks reproduzierbarkeit
    #logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    # ---------------------------------------------------- Userinput für Methoden ----------------------------------------------------
    while True:
        vectmethod = input("Type 0 for BoW or 1 for Tf-idf: ")
        if vectmethod in ('0', '1'):
            break
    
    while True:
        findnumtop = input("Type 0 to find most relevant topics or 1 to find optimal number of topics: ")
        if findnumtop in ('0', '1'):
            break
    
    if findnumtop == "0":
        while True:
            extrmethod = input("Type 0 for LDA, 1 for LSI or 2 for HDP: ")
            if extrmethod in ('0', '1', '2'):
                break
    else:
        while True:
            extrmethod = input("Type 0 for LDA or 1 for LSI: ")
            if extrmethod in ('0', '1'):
                break

    # ---------------------------------------------------- Daten aus JSON einlesen ----------------------------------------------------
    print("Loading data")
    df = pd.read_json("yelp_academic_dataset_review.json", lines=True, nrows=50000) # load table from json into dataframe
    df = df[df.stars < 2] # remove 2, 3, 4 and 5 star reviews
    docs = df.text # load data from "text" column into list
    print(len(docs), "documents loaded")

    # ---------------------------------------------------- Preprocessing ----------------------------------------------------
    print("Preprocessing documents")
    def preprocess_docs(text): # function for preprocessing
        text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "", text) # remove urls
        text = re.sub(r"\S+@\S+", "", text) # remove emails
        text = re.sub(r"(\d+),(\d+),?(\d*)", "", text) # remove numbers in format xxxx,xxxx
        text = re.sub(r"(\+)?\d+[ ]?\d*[ ]?\d*[ ]?\d*", "", text) # remove numbers in all other formats
        text = text.strip() # remove leading and trailing white space
        text = " ".join(text.split()) # replace multiple consecutive white space characters with a single space
        tokens = nltk.word_tokenize(text) # tokenize the text
        lower = [token.lower() for token in tokens] # lowercase the tokens
        filtered = [token for token in lower if token not in string.punctuation] # remove punctuation
        stopwords = nltk.corpus.stopwords.words("english") # get list of stopwords in English
        #stopwords.extend(["example"]) # extend list of stopwords with custom words
        filtered2 = [token for token in filtered if token not in stopwords] # remove stopwords
        filtered3 = [token for token in filtered2 if len(token) > 2] # remove words with less than 3 characters
        lemmatizer = nltk.WordNetLemmatizer() # create lemmatizer object
        lemmas = [lemmatizer.lemmatize(token, get_pos(token)) for token in filtered3] # lemmatize each token
        return lemmas

    def get_pos(word): # Funktion für nltk pos zu wordnet pos mapping
        pos = nltk.pos_tag([word]) # get the nltk pos tag 
        pos = pos[0][1][0] # get the first letter of the nltk pos tag
        pos_tags = {
            "N": nltk.corpus.wordnet.NOUN,
            "V": nltk.corpus.wordnet.VERB,
            "J": nltk.corpus.wordnet.ADJ,
            "R": nltk.corpus.wordnet.ADV,
            "default": nltk.corpus.wordnet.NOUN
        }
        return pos_tags.get(pos, pos_tags['default']) # if the pos tag doesnt match any key, return wordnet.NOUN as a default

    nltk.download('averaged_perceptron_tagger_eng', quiet=True) # Für pos tagging benötigt
    processed_docs = [preprocess_docs(doc) for doc in docs]

    # ---------------------------------------------------- Vectorizing ----------------------------------------------------
    print("Startig vectorizing process")
    dictionary = gensim.corpora.Dictionary()
    dictionary.filter_extremes(no_below=50, no_above=0.5) #--------------
    #dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000) #--------------
    bow_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in processed_docs]
    if vectmethod == "0":
        docname = "BoW_"
        vect = bow_corpus
    else:
        docname = "Tf-idf_"
        tfidf = gensim.models.TfidfModel(bow_corpus)
        tfidf_corpus = tfidf[bow_corpus]
        vect = tfidf_corpus
    #print(dictionary)

    # ---------------------------------------------------- Topic Modeling ----------------------------------------------------
    print("Modeling topics")
    # parameters for finding optimal amount of topics
    limit=101
    start=1
    step=1
    
    if ((findnumtop == "0") and (extrmethod == "0")):
        docname += "LDA.txt"
        model =  gensim.models.LdaModel(corpus=vect, id2word=dictionary, num_topics=100, alpha="auto", eta="auto", random_state=100, chunksize=10000, passes=100)
        print(model.print_topics())
        f = open(docname, "w")
        f.write(model.print_topics())
        f.close()

    elif ((findnumtop == "0") and (extrmethod == "1")):
        docname += "LSI.txt"
        model = gensim.models.lsimodel.LsiModel(corpus=vect, id2word=dictionary, num_topics=10, chunksize=10000)
        print(model.print_topics())
        f = open(docname, "w")
        f.write(model.print_topics())
        f.close()

    elif  ((findnumtop == "0") and (extrmethod == "2")):
        docname += "HDP.txt"
        model = gensim.models.hdpmodel.HdpModel(corpus=vect, id2word=dictionary) # no need to specify the numebr of topics as the HDP model learns the best suitable number of topics on its own
        print(model.print_topics())
        f = open(docname, "w")
        f.write(model.print_topics())
        f.close()

    elif ((findnumtop == "1") and (extrmethod == "0")):
        docname += "LDA_NumTopics.txt"
        def calculate_coherence(corpus, dictionary, texts, start, limit, step):
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                print("Calculating coherence for", num_topics, "topics")
                model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha="auto", eta="auto", random_state=100, chunksize=10000, passes=100)
                model_list.append(model)
                coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())
            return model_list, coherence_values
        
        model_list, coherence_values = calculate_coherence(vect, dictionary, processed_docs, start, limit, step)

        f = open(docname, "w")
        x = range(start, limit, step)
        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of ", round(cv, 4))
            f.write("Num Topics = %s has Coherence Value of %s\n" % (m, round(cv, 4)))
        f.close()

    else:
        docname += "LSI_NumTopics.txt"
        def calculate_coherence(corpus, dictionary, texts, start, limit, step):
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                print("Calculating coherence for", num_topics, "topics")
                model = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_seed=100, chunksize=10000, power_iters=100)
                model_list.append(model)
                coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())
            return model_list, coherence_values
        
        model_list, coherence_values = calculate_coherence(vect, dictionary, processed_docs, start, limit, step)

        f = open(docname, "w")
        x = range(start, limit, step)
        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of ", round(cv, 4))
            f.write("Num Topics = %s has Coherence Value of %s\n" % (m, round(cv, 4)))
        f.close()