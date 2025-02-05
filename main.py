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
            if extrmethod in ('0', '1'):
                while True:
                    ntopics = int(input("Define the number of topics: "))
                    if 0 < ntopics <= 50:
                        break
                break
            elif extrmethod in ('2'):
                break

    else:
        while True:
            extrmethod = input("Type 0 for LDA or 1 for LSI: ")
            if extrmethod in ('0', '1'):
                break

    # ---------------------------------------------------- Daten aus JSON laden ----------------------------------------------------
    print("Loading data")

    review_df = pd.read_json("yelp_academic_dataset_review.json", lines=True, nrows=1000000) # load review table from json into dataframe
    business_df = pd.read_json("yelp_academic_dataset_business.json", lines=True) # load business table from json into dataframe
    merged_df = pd.merge(review_df, business_df, on='business_id') # merge the two dataframes by the id of the business
    filtered_df = merged_df[
        (merged_df['categories'].str.contains('Hotel', na=False)) & # filter out all reviews of businesses which are not hotels
        (merged_df['stars_x'] == 1) # only retain all 1 star reviews
    ]
    docs = filtered_df.text # load data from "text" column into list
    
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
        stopwords.extend(["star", "stars"]) # extend list of stopwords with custom words
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

    nltk.download('averaged_perceptron_tagger', quiet=True) # required for pos tagging
    processed_docs_temp = [preprocess_docs(doc) for doc in docs] # preprocess documents
    processed_docs = [doc for doc in processed_docs_temp if doc] # remove documents which could have gotten empty after preprocessing

    print((len(processed_docs_temp) - len(processed_docs)), "empty documents removed")

    # create bigrams and add them to corpus
    bigram = gensim.models.Phrases(processed_docs, min_count=50, threshold=45)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    processed_docs = [bigram_mod[doc] for doc in processed_docs]

    # ---------------------------------------------------- Vectorizing ----------------------------------------------------
    print("Vectorizing tokens")
    dictionary = gensim.corpora.Dictionary(processed_docs) # create dictionary object and populate it with tokens from processed documents
    dictionary.filter_extremes(no_below=50, no_above=0.33) # filter out tokens that appear in only a few documents or in a lot
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs if len(doc) > 0] # load tokens into corpus and dictionary

    print("Number of unique tokens in dictionary:", len(dictionary))

    if vectmethod == "0":
        docname = "BoW_"
        vect = bow_corpus
    else:
        docname = "Tf-idf_"
        tfidf = gensim.models.TfidfModel(bow_corpus) # create tf-idf model
        tfidf_corpus = tfidf[bow_corpus] # transform bow corpus to tf-idf corpus
        vect = tfidf_corpus

    # ---------------------------------------------------- Topic Modeling ----------------------------------------------------
    print("Modeling topics")
    # range used for finding optimal amount of topics based on their respective coherence score
    limit=51
    start=1
    step=1

    if (findnumtop == "0"):
        if (extrmethod == "0"):
            docname += "LDA.txt"
            model =  gensim.models.LdaMulticore(corpus=vect, id2word=dictionary, num_topics=ntopics, workers=3, passes=100, chunksize=2000) #create lda model

        elif (extrmethod == "1"):
            docname += "LSI.txt"
            model = gensim.models.LsiModel(corpus=vect, id2word=dictionary, num_topics=ntopics) #create lsi model

        elif  (extrmethod == "2"):
            docname += "HDP.txt"
            model = gensim.models.HdpModel(corpus=vect, id2word=dictionary, chunksize=2000) #create hdp model; number of topics is not needed since hdp determines the optimal amount by itself

        coherencemodel = gensim.models.CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v') # create coherence model
        coherencemodel2 = gensim.models.CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='u_mass') # create coherence model
        cv = round(coherencemodel.get_coherence(), 4) # calculate coherence
        cv2 = round(coherencemodel2.get_coherence(), 4) # calculate coherence
        # write coherence and topics of the model to file
        f = open(docname, "w")
        f.write("Model has Coherence of c_v %s and u_mass %s\n" % (cv, cv2))
        for topic in model.print_topics(num_words=10):
            f.write("{0}\n".format(topic))
        f.close()

    else:
        if (extrmethod == "0"):
            docname += "LDA_NumTopics.txt"
        else:
            docname += "LSI_NumTopics.txt"
        PYTHONHASHSEED=0 # needed to reproduce results
        def calculate_coherence(corpus, dictionary, texts, start, limit, step): # create a model for each number of topics in range and calculate their respective coherence
            coherence_values = []
            coherence_values2 = []
            model_list = []
            for num_topics in range(start, limit, step): # loop for defined range
                print("Calculating coherence for", num_topics, "topics")
                if (extrmethod == "0"):
                    model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=3, random_state=42, passes=200, chunksize=1000) #create lda model; random_state is defined for reproducibility
                else:
                    model = gensim.models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_seed=100) #create lsi model; random_seed is defined for reproducibility
                model_list.append(model)
                coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v') # create coherence model
                cv = round(coherencemodel.get_coherence(), 4) # calculate coherence
                coherencemodel2 = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass') # create coherence model
                cv2 = round(coherencemodel2.get_coherence(), 4) # calculate coherence
                print("Model has Coherence of c_v %s and u_mass %s" % (cv, cv2))
                coherence_values.append(cv)
                coherence_values2.append(cv2)
            return model_list, coherence_values, coherence_values2
        model_list, coherence_values, coherence_values2 = calculate_coherence(vect, dictionary, processed_docs, start, limit, step)
        # write results to file
        f = open(docname, "w")
        x = range(start, limit, step)
        for m, cv, cv2 in zip(x, coherence_values, coherence_values2):
            f.write("Num Topics = %s has Coherence of c_v %s and u_mass %s\n" % (m, cv, cv2))
        f.close()