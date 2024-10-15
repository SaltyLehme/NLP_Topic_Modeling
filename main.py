import pandas as pd
import re
import nltk
import string
import gensim

if __name__ == "__main__":
    # ---------------------------------------------------- Daten aus JSON einlesen ----------------------------------------------------
    df = pd.read_json("data.json", lines=True) # load table from json into dataframe
    df = df[df.stars < 2] # remove 3, 4 and 5 star reviews
    docs = df.text # load data from "text" column into list
    print(len(docs))

    # ---------------------------------------------------- Preprocessing ----------------------------------------------------
    def preprocess_docs(text): # function for preprocessing
        pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?" # define a regular expression pattern to match URLs
        text = re.sub(pattern, "", text) # remove urls
        text = text.strip() # remove leading and trailing white space
        text = " ".join(text.split()) # replace multiple consecutive white space characters with a single space
        tokens = nltk.word_tokenize(text) # tokenize the text
        lower = [token.lower() for token in tokens] # lowercase the tokens
        filtered = [token for token in lower if token not in string.punctuation] # remove punctuation
        stopwords = nltk.corpus.stopwords.words("english") # get list of stopwords in English
        #stopwords.extend(["example"]) # extend list of stopwords with custom words
        filtered2 = [token for token in filtered if token not in stopwords] # remove stopwords
        lemmatizer = nltk.WordNetLemmatizer() # create lemmatizer object
        lemmas = [lemmatizer.lemmatize(token) for token in filtered2] # lemmatize each token
        return lemmas

    processed_docs = [preprocess_docs(doc) for doc in docs]

    # ---------------------------------------------------- Vectorizing ----------------------------------------------------
    vectmethod = input("Type 0 for BoW or 1 for Tf-idf: ")
    dictionary = gensim.corpora.Dictionary()
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000) #--------------
    bow_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in processed_docs]
    if vectmethod == "0":
        vect = bow_corpus
    else:
        tfidf = gensim.models.TfidfModel(bow_corpus)
        tfidf_corpus = tfidf[bow_corpus]
        vect = tfidf_corpus

    # ---------------------------------------------------- Extraction ----------------------------------------------------
    extrmethod = input("Type 0 for LDA, 1 for LSI, 2 for HDP or 3 to find optimal number of topics for LDA: ")
    if extrmethod == "0":
        model =  gensim.models.LdaMulticore(vect, dictionary, num_topics = 10, passes = 100, workers = 2)
        #model =  gensim.models.LdaMulticore(vect, dictionary, num_topics=20, workers=2, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        #print(model.print_topics())

    if extrmethod == "1":
        model = gensim.models.lsimodel.LsiModel(vect, dictionary, num_topics=10, chunksize=100)
        #print(model.print_topics())

    if extrmethod == "2":
        model = gensim.models.hdpmodel.HdpModel(vect, dictionary)
        #print(model.print_topics())

    if extrmethod == "3":
        def calculate_coherence(corpus, dictionary, texts, limit, start=2, step=2):
            """
            Compute c_v coherence for various number of topics

            Parameters:
            ----------
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            limit : Max num of topics

            Returns:
            -------
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
            """
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                model = gensim.models.LdaMulticore(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                passes=100,
                                                workers=2)
                model_list.append(model)
                coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())

            return model_list, coherence_values
        
        model_list, coherence_values = calculate_coherence(vect, dictionary, processed_docs, start=2, limit=5, step=2)

        limit=5; start=2; step=2;
        x = range(start, limit, step)

        # Print the coherence scores
        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    print(model.print_topics())

"""
    for idx, topic in model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic ))
        print("\n")
        """