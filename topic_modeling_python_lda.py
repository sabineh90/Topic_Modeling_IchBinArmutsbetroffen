#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sabineherberth

"""

# =============================================================================
# Import packages 
# =============================================================================

# Base and Cleaning
import pandas as pd
import re
import unicodedata
from pprint import pprint

# Natural Language Processing (NLP)
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Topic Modeling
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Visualization
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt


# =============================================================================
# Setting file paths of tweets
# =============================================================================

root = '/Users/sabineherberth/Documents/02_UNI/Master/Faecher/Masterarbeit/'
path_tweets = root + 'Twitter_Data/finales_Datenset/tweets_ichbinarmutsbetroffen161222_ohneRT_final.csv'
output = root + 'Output_LDA/'


# =============================================================================
# Data Loading
# =============================================================================

def load_data(path_tweets):
    return pd.read_csv(path_tweets)
Tweet_df=load_data(path_tweets)
print(Tweet_df)


# =============================================================================
# Drop Duplicates
# =============================================================================

#drop duplicates and keep only first version of the tweet
Tweet_df.drop_duplicates(subset=['text'], inplace=True, keep='first')


# =============================================================================
# Data Cleaning
# =============================================================================

#converting every row of the column into lower case 
def to_lowercase(text):
    return text.lower()

#creating new column in dataframe with cleaned data
Tweet_df['cleaned_data'] = Tweet_df.text.apply(to_lowercase)


#standardizing accented characters for every row and removing emojis
def standardize_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

Tweet_df.cleaned_data=Tweet_df.cleaned_data.apply(standardize_accented_chars)


#removing urls from every row
def remove_url(text):
    text = re.sub(r'https?:\S*', '', text)
    text = re.sub(r'bit.ly?:\S+', '', text)
    return text

Tweet_df.cleaned_data=Tweet_df.cleaned_data.apply(remove_url)


#removing mentions and tags from every row
def remove_mentions (text):
    text = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', text) # remove retweet
    text = re.sub(r'(@[A-Za-z]+[A-Za-z0-9-_]+)', '', text) # remove tweeted at
    return text

Tweet_df.cleaned_data=Tweet_df.cleaned_data.apply(remove_mentions)


#keeping only alphabet
def keep_only_alphabet(text):
    return re.sub(r'[^a-z]', ' ', text)

Tweet_df.cleaned_data=Tweet_df.cleaned_data.apply(keep_only_alphabet)


# =============================================================================
# Removing stopwords and short words 
# =============================================================================

def remove_stopwords(text,nlp,custom_stop_words=None,remove_small_tokens=True,min_len=2):
#if custom stopwords are provided, add them to the default stopwords list 
    if custom_stop_words:
        nlp.Defaults.stop_words |= custom_stop_words
    
    filtered_sentence =[] 
    doc=nlp (text)
    for token in doc:
        
        if token.is_stop == False: 
            
            if remove_small_tokens:
                if len(token.text)>min_len:
                    filtered_sentence.append(token.text)
            else:
                filtered_sentence.append(token.text) 
                
    if len(filtered_sentence) > 0:
        return " ".join(filtered_sentence)
    else:
        return None
    
#creating a spaCy object. 
nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])

#removing stop-words and short words from every row
Tweet_df["no_stopwords"]=Tweet_df.cleaned_data.apply(lambda x:remove_stopwords(x,nlp,{"ichbinarmutsbetroffen", "ichbinarmutbetroffen", 
                                                                                   "none", "hashtag", "armut", "fur", "amp", "mal", 
                                                                                   "menschen", "link", "uber", "evtl", "auch", "thread" }))


# =============================================================================
# Data Preprocessing
# =============================================================================

def clean_text(text):

    """
        1) Tokenising the text
        2) Keeping only nouns, adjectives and adverbs
        3) Lemmatising the words
        4) Stemming the words
        5) Removing all words with less than three characters 
        6) Removing all "sentences" that consist of only one word
    
    """    

# Typenliste für Wörter erstellen:

    pos_types = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

# Stemmer laden

    ps = PorterStemmer()

    wnl = WordNetLemmatizer()

    t = nltk.tokenize.word_tokenize(text)

    t = [word[0] for word in nltk.pos_tag(t) if word[1] in pos_types]
    t = [wnl.lemmatize(word) for word in t]
    t = [ps.stem(word) for word in t]
    t = [word for word in t if len(word) > 1]

    return(t)

Tweet_df.no_stopwords=Tweet_df.no_stopwords.astype(str)

#storing the generated tokens in a new column named 'tokens'
Tweet_df['tokens']=Tweet_df.no_stopwords.apply(clean_text)

#make tokens a string again in new column
Tweet_df['tokens_back_to_text']=Tweet_df.tokens.astype(str)


# =============================================================================
# Dictionary
# =============================================================================

def create_dictionary(words):
    return corpora.Dictionary(words)

#passing the dataframe column having tokens as the argument
id2word=create_dictionary(Tweet_df.tokens)
print(id2word)

print("Anzahl der Tokens im Lexikon vor der Bereinigung:", len(id2word))
id2word.filter_extremes(no_below=5, no_above=0.9, keep_n=50000)
print("Anzahl der Tokens im Lexikon nach der Bereinigung:", len(id2word))


def create_document_matrix(tokens,id2word):
    corpus = []
    for text in tokens:
        corpus.append(id2word.doc2bow(text))
    return corpus
    
#passing the dataframe column having tokens and dictionary
corpus=create_document_matrix(Tweet_df.tokens,id2word)
print(Tweet_df.tokens[0:10])
print(corpus[0:10])

#(word_ID, word_frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[0:10]]



# =============================================================================
# Topic Modeling
# =============================================================================

# =============================================================================
# Base Model
# =============================================================================

# Instantiating a Base LDA model 

base_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                             id2word=id2word,
                                             num_topics=20,
                                             random_state=100,
                                             update_every=1,
                                             chunksize=100,
                                             passes=10,
                                             alpha='auto',
                                             per_word_topics=True)
    
# Print Keywords

pprint(base_model.print_topics())
doc_lda = base_model[corpus]

# Compute Perplexity
base_perplexity = base_model.log_perplexity(corpus)
print('\nPerplexity: ', base_perplexity) 

# Compute Coherence Score
coherence_model = CoherenceModel(model=base_model, texts=Tweet_df['tokens'], 
                                   dictionary=id2word, coherence='c_v')
coherence_lda_model_base = coherence_model.get_coherence()
print('\nCoherence Score: ', coherence_lda_model_base)



#Creating Topic Distance Visualization 
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(base_model, corpus, id2word)

pyLDAvis.display(vis)

pyLDAvis.save_html(vis, "interaktive_Grafik_1" + str(20) + "_Themen.html")


# =============================================================================
# Hyperparameter Tuning 
# =============================================================================

# =============================================================================
# Grid Search
# =============================================================================

vectorizer = CountVectorizer()
data_vectorized = vectorizer.fit_transform(Tweet_df['tokens_back_to_text'])

# Define Search Param
search_params = {'n_components': [10, 20, 30, 40, 50, 60, 70], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)
GridSearchCV(cv=None, error_score='raise',
             estimator=LatentDirichletAllocation(batch_size=128, 
                                                 doc_topic_prior=None,
                                                 evaluate_every=-1, 
                                                 learning_decay=0.7, 
                                                 learning_method=None,
                                                 learning_offset=10.0, 
                                                 max_doc_update_iter=100, 
                                                 max_iter=10,
                                                 mean_change_tol=0.001, 
                                                 n_components=10, 
                                                 n_jobs=1,
                                                 perp_tol=0.1, 
                                                 random_state=None,
                                                 topic_word_prior=None, 
                                                 total_samples=1000000.0, 
                                                 verbose=0),
             n_jobs=1,
             param_grid={'n_topics': [10, 20, 30, 40, 50, 60, 70], 
                         'learning_decay': [0.5, 0.7, 0.9]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
             scoring=None, verbose=0)

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))


# =============================================================================
# Optimum number of topics
# =============================================================================

#Defining a function to loop over number of topics to be used to find an optimal number of topics

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
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
    coherence_values : Coherence values corresponding to the 
    LDA model with respective number of topics
    """
    coherence_values_topic = []
    model_list_topic = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list_topic.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_topic.append(coherencemodel.get_coherence())

    return model_list_topic, coherence_values_topic    
     

# Can take a long time to run.
model_list_topic, coherence_values_topic = compute_coherence_values(dictionary=id2word,
                                                        corpus=corpus,
                                                        texts=Tweet_df['tokens'],
                                                        start=2, limit=130, step=6)

# Show graph for choosing optimal model with coherence scores

limit=130; start=2; step=6
x = range(start, limit, step)
plt.plot(x, coherence_values_topic)
plt.xlabel("Num Topics")
plt.ylabel("Coherence Score")
plt.legend(("coherence_values_topic"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values_topic):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
#pick the model that gave the highest CValue before flattening out
#select the model and print the topics 


# =============================================================================
# Finding the best model
# =============================================================================

# testing 26 models with different variables and find model with highest scores
# tested variables:
# num_topics: 74, 98
# passes: 10, 15, 20, 25
# alpha: symmetric, asymmetric
# decay: 0.5, 0.7, 0.9
# optimal number of iterations: 50, 60, 70, 80, 90, 100
# minimum probability 0.01 - 0.1


topic_model_1 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=74,
                       random_state=42,
                       chunksize=2000,
                       passes=10,
                       )
    
# Compute Perplexity

model1_perplexity = topic_model_1.log_perplexity(corpus)
print('\nPerplexity: ', model1_perplexity) 

# Compute Coherence Score
coherence_model = CoherenceModel(model=topic_model_1, texts=Tweet_df['tokens'], 
                                   dictionary=id2word, coherence='c_v')
coherence_lda_model_1 = coherence_model.get_coherence()
print('\nCoherence Score: ', coherence_lda_model_1)


# =============================================================================
# Final Topic Model
# =============================================================================

final_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=40,
                       random_state=42,
                       chunksize=2000,
                       passes=10,
                       decay=0.5,
                       iterations=60, 
                       )

# Filtering for words 
words = [re.findall(r'"([^"]*)"',t[1]) for t in final_model.print_topics()]

# Create Topics
topics = [' '.join(t[0:10]) for t in words]

# Getting the topics
for id, t in enumerate(topics): 
    print(f"------ Topic {id} ------")
    print(t, end="\n\n")


# =============================================================================
# Export Topics to Excel
# =============================================================================

model_topics = final_model.print_topics(num_topics = 40, num_words=10)
model_topics = pd.DataFrame([x[1] for x in model_topics], columns=["ladungen_tokens"])
model_topics.index.name = "topic_number"
#model_topics.to_excel(output + "topic_token" + str(98) + ".xlsx")

#keeping only alphabet
def keep_only_alphabet(text):
    return re.sub(r'[^a-z]', ' ', text)

#for all the rows
model_topics["only_tokens"]=model_topics.ladungen_tokens.apply(keep_only_alphabet)
model_topics.to_excel(output + "topic_tokens_40" + ".xlsx")


# =============================================================================
# Visualization 
# =============================================================================

#Creating Topic Distance Visualization 
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(final_model, corpus, id2word)

pyLDAvis.display(vis)

pyLDAvis.save_html(vis, "interaktive_Grafik" + str(40) + "_Themen.html")


# =============================================================================
# Finding the dominant topic in each document
# =============================================================================

def format_topics_document(ldamodel=final_model, corpus=corpus, texts=Tweet_df.text):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_document(ldamodel=final_model, corpus=corpus, texts=Tweet_df.text)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)

df_dominant_topic.to_excel(output + "dominant_topic_per_sentence_tweets" + ".xlsx")


# =============================================================================
# Find the most representative tweet for each topic
# =============================================================================

# Group top 5 sentences under each topic
df_repres_tweet_per_topic = pd.DataFrame()

df_topic_sents_keywords = format_topics_document(ldamodel=final_model, corpus=corpus, texts=Tweet_df.text)
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    df_repres_tweet_per_topic = pd.concat([df_repres_tweet_per_topic, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
df_repres_tweet_per_topic.reset_index(drop=True, inplace=True)

# Format
df_repres_tweet_per_topic.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
df_repres_tweet_per_topic.head()

df_repres_tweet_per_topic.to_excel(output + "repres_tweet_per_topic3" + ".xlsx")


# =============================================================================
# Topic distribution across documents
# =============================================================================
#welches Topic kommt in den meisten Tweets vor?

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

df_dominant_topics.to_excel(output + "dominant_topics" + ".xlsx")











































