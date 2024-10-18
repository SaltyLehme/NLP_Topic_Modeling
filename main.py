import pandas as pd
import re
import nltk
import string
import gensim

if __name__ == "__main__":
    # ---------------------------------------------------- Userinput für zu verwendende Methoden ----------------------------------------------------
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

    # ---------------------------------------------------- Daten aus JSON laden ----------------------------------------------------
    print("Loading data")
    df = pd.read_json("yelp_academic_dataset_review.json", lines=True, nrows=5000) # load table from json into dataframe
    df = df[df.stars < 2] # remove 2, 3, 4 and 5 star reviews
    docs = df.text # load data from "text" column into list
    print(len(docs), "documents loaded")

    # ---------------------------------------------------- Preprocessing ----------------------------------------------------
    print("Preprocessing documents")
    def preprocess_docs(text): # function for preprocessing
        text = re.sub(r"(\<[ ]?[a-z]+>|\<\/[a-z]+\>)", "", text) # remove html
        text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "", text) # remove urls
        text = re.sub(r"\S+@\S+", "", text) # remove emails
        text = re.sub(r"(\d+),(\d+),?(\d*)", "", text) # remove numbers in format xxxx,xxxx
        text = re.sub(r"(\+)?\d+[ ]?\d*[ ]?\d*[ ]?\d*", "", text) # remove numbers in all other formats
        text = re.sub(r"(\.\.\.)", "", text) # remove ...
        text = text.strip() # remove leading and trailing white space
        text = " ".join(text.split()) # replace multiple consecutive white space characters with a single space
        tokens = nltk.word_tokenize(text) # tokenize the text
        lower = [token.lower() for token in tokens] # lowercase the tokens
        filtered = [token.strip(string.punctuation) for token in lower] # remove punctuation
        stopwords = nltk.corpus.stopwords.words("english") # get list of stopwords in English
        #stopwords.extend(["example"]) # extend list of stopwords with custom words
        filtered2 = [token for token in filtered if token not in stopwords] # remove stopwords
        filtered3 = [token for token in filtered2 if len(token) > 2] # remove words with less than 3 characters
        lemmatizer = nltk.WordNetLemmatizer() # create lemmatizer object
        lemmas = [lemmatizer.lemmatize(token, get_pos(token)) for token in filtered3] # lemmatize each token
        return lemmas

    def get_pos(word): # function for nltk pos to wordnet pos mapping
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

    nltk.download('averaged_perceptron_tagger_eng', quiet=True) # required for pos tagging
    processed_docs = [preprocess_docs(doc) for doc in docs] # preprocess documents

    # create bigrams and add them to corpus
    bigram = gensim.models.Phrases(processed_docs, min_count=10)
    for idx in range(len(processed_docs)):
        for token in bigram[processed_docs[idx]]:
            if '_' in token:
                processed_docs[idx].append(token)

    # ---------------------------------------------------- Vectorizing ----------------------------------------------------
    print("Vectorizing tokens")
    dictionary = gensim.corpora.Dictionary() # create dictionary object
    #dictionary.filter_extremes(no_below=50, no_above=0.5) # filter out tokens that appear in only a few documents or in a lot
    bow_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in processed_docs] # load tokens into corpus and dictionary
    if vectmethod == "0":
        docname = "BoW_"
        vect = bow_corpus
    else:
        docname = "Tf-idf_"
        tfidf = gensim.models.TfidfModel(bow_corpus) # create tf-idf model
        tfidf_corpus = tfidf[bow_corpus] # transform bow corpus to tf-idf corpus
        vect = tfidf_corpus
    #print(dictionary)

    # ---------------------------------------------------- Topic Modeling ----------------------------------------------------
    print("Modeling topics")
    # range used for finding optimal amount of topics based on their respective coherence score
    limit=101
    start=1
    step=1

    if ((findnumtop == "0") and (extrmethod == "0")):
        docname += "LDA.txt"
        model =  gensim.models.LdaMulticore(corpus=vect, id2word=dictionary, num_topics=10, workers=3, passes=100, chunksize=2000) #create lda model
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v') # create coherence model
        cv = round(coherencemodel.get_coherence(), 4) # calculate coherence
        # write coherence and topics of the model to file
        f = open(docname, "w")
        f.write("Model has Coherence Value of %s\n" % (cv))
        for topic in model.print_topics(num_words=10):
            f.write("{0}\n".format(topic))
        f.close()

    elif ((findnumtop == "0") and (extrmethod == "1")):
        docname += "LSI.txt"
        model = gensim.models.LsiModel(corpus=vect, id2word=dictionary, num_topics=2) #create lsi model
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v') # create coherence model
        cv = round(coherencemodel.get_coherence(), 4) # calculate coherence
        # write coherence and topics of the model to file
        f = open(docname, "w")
        f.write("Model has Coherence Value of %s\n" % (cv))
        for topic in model.print_topics():
            f.write("{0}\n".format(topic))
        f.close()

    elif  ((findnumtop == "0") and (extrmethod == "2")):
        docname += "HDP.txt"
        model = gensim.models.HdpModel(corpus=vect, id2word=dictionary, chunksize=2000) #create hdp model; number of topics is not needed since hdp determines the optimal amount by itself
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v') # create coherence model
        cv = round(coherencemodel.get_coherence(), 4) # calculate coherence
        # write coherence and topics of the model to file
        f = open(docname, "w")
        f.write("Model has Coherence Value of %s\n" % (cv))
        for topic in model.print_topics():
            f.write("{0}\n".format(topic))
        f.close()

    elif ((findnumtop == "1") and (extrmethod == "0")):
        PYTHONHASHSEED=0 # needed to reproduce results
        docname += "LDA_NumTopics.txt"
        def calculate_coherence(corpus, dictionary, texts, start, limit, step): # create a lda model for each number of topics in range and calculate their respective coherence
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step): # loop for defined range
                print("Calculating coherence for", num_topics, "topics")
                model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=3, random_state=100, passes=100, chunksize=1000) #create lda model; random_state is defined for reproducibility
                model_list.append(model)
                coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v') # create coherence model
                cv = round(coherencemodel.get_coherence(), 4) # calculate coherence
                print("Model has Coherence Value of", cv)
                coherence_values.append(cv)
            return model_list, coherence_values
        model_list, coherence_values = calculate_coherence(vect, dictionary, processed_docs, start, limit, step)
        # write results to file
        f = open(docname, "w")
        x = range(start, limit, step)
        for m, cv in zip(x, coherence_values):
            f.write("Num Topics = %s has Coherence Value of %s\n" % (m, cv))
        f.close()

    else:
        PYTHONHASHSEED=0 # needed to reproduce results
        docname += "LSI_NumTopics.txt"
        def calculate_coherence(corpus, dictionary, texts, start, limit, step): # create a lsi model for each number of topics in range and calculate their respective coherence
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step): # loop for defined range
                print("Calculating coherence for", num_topics, "topics")
                model = gensim.models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_seed=100) #create lda model; random_seed is defined for reproducibility
                model_list.append(model)
                coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v') # create coherence model
                cv = round(coherencemodel.get_coherence(), 4) # calculate coherence
                print("Model has Coherence Value of", cv)
                coherence_values.append(cv)
            return model_list, coherence_values
        model_list, coherence_values = calculate_coherence(vect, dictionary, processed_docs, start, limit, step)
        # write results to file
        f = open(docname, "w")
        x = range(start, limit, step)
        for m, cv in zip(x, coherence_values):
            f.write("Num Topics = %s has Coherence Value of %s\n" % (m, cv))
        f.close()