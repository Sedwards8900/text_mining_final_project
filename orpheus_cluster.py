# %%
# Only run if you need specific libraries when using jupyter
# !pip install pandas
# !pip install numpy
# !pip install spacy
# !pip install demoji
# !pip install fasttext
# !pip install requests
# !pip install tqdm
# !pip install sklearn
# !pip install "gensim>=4.0.0"

# %%
# Imports

# Data access, editing and management
import pandas as pd
import numpy as np
import sqlite3
# POS extraction and tokenization
import spacy
import regex as re
import demoji
# Language identification for tokens
import fasttext
import requests
# Processing time visualization
from tqdm import tqdm
tqdm.pandas()
# To run console commands for installation
import os
# For vectorization and clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# %% [markdown]
# # Extract DF from part 1

# %%
# Extracting articles df
df = pd.read_pickle('articles.pkl')
# df

# %% [markdown]
# # Filter DF to only show as much relevant data as possible, eliminating files with errors, empty strings, and no content

# %%
# Filter out bad request or any empty body data on the df

# Remove articles that do not have any text - run only once to not cause error
df = df.loc[df['body_text'] != '']
df.loc[df['body_text'] == '']

# %%
# Get rid of any nan values in body and title
a = df.loc[~df['title'].isnull()]
df = a
b = a.loc[~a['body_text'].isnull()]
df = b

# %%
df.loc[df['body_text'].isnull()]

# %%
# Remove any 403 error files
a = df.loc[(~df['title'].str.contains('403')) & (~df['url'].isnull())]
b = a.loc[(~a['title'].str.contains('Forbidden')) & (~a['url'].isnull())]
df = b
b.loc[df['title'].str.contains('Forbidden')]

# %%
# Reset index after performing first cleanup
df = df.reset_index(drop=True)

# %% [markdown]
# # Clean up text data to ensure POS tokens significance can be extracted properly by spacy

# %%
# Get emoji codes from library demoji
demoji.download_codes()

# %%
'''
Function to remove any emojis in text
'''
def remove_emojis(text):
    dem = demoji.findall(text)
    for item in dem.keys():
        text = re.sub(item, '', text)
    return text

# %%
# Function to clean unprocessed text body from each article
def clean_text(text):
    # Remove emojis
    text = remove_emojis(text)
    # Remove any solo numbers in line
    text = re.sub(r'^[0-9]+\s.*', '', text, flags=re.MULTILINE)
    # Replace multiple spaces with one
    text = re.sub(' +', ' ', text)
    # Replace multiple newlines by splitting string into substrings saved in a list
    text = re.split('\s*\n\s*\n', text)
    text = [item for item in text if len(re.split(' ', item)) > 5]
    # Join all vals in list as a one string again
    text = " ".join(text)
    # Replace newline characters left with single space
    text = re.sub('\n', ' ', text)
    # Return cleaned text string
    return text

# %%
# Clean text within column 'body_text' in dataframe df
df['body_text'] = [clean_text(text) for text in df['body_text']]

# %%
df.head(3)

# %% [markdown]
# ## Store df as pickle - intermediary step

# %%
df.to_pickle('articles_cleaned.pkl')

sql = sqlite3.connect('articles_cleaned.db')
df.to_sql('articles_cleaned', sql, if_exists='replace')
sql.close()

# %% [markdown]
# # Tokenization and POS tagging via Spacy

# %% [markdown]
# ## Language detection function to ensure all tokens will be properly tagged based on the english language via fasttext

# %%
# Download from web the model for language detection from fasttext
model_filename = "lid.176.ftz"
r = requests.get(f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{model_filename}")
open(model_filename, 'wb').write(r.content)
# load language identification model 
lang_model = fasttext.load_model(model_filename)

# %%
'''
Function that detects if a text contains a great quantity of tokens in another language,
if it does, it is removed from our sample, but if it is below the threshold, we do not remove
the words due to the fact that natural disasters often use name of locations in the U.S. that
are rooted in words from other languages.

'''
def detect_language(doc):
    # Temporarily remove newline symbols due to function format requirements
    text = doc.replace("\n", " ")
    # Set desired language
    correct = "en"
    doc_l = lang_model.predict(text, k=2)
    # If first label is english and english label is highest value of two found languages
    if (doc_l[0][0].replace("__label__", "") == "en" and doc_l[1][0] > 2*doc_l[1][1]) or (doc_l[0][1].replace("__label__", "") == "en" and doc_l[1][1] > doc_l[1][0]):
        # Confirm to use text
        return True

        # In case we wish to proceed to evaluate tokens and extract unnecessary non-english words
        # # Create flag
        # ok = not_ok = 0
        # # Perform individual token checks via fasttext
        # for token in tokens:
        #     l = lang_model.predict(token, k=2)
        #     # If percentage in english is greater than twice the percentage of other language
        #     if l[1][0] > 2*l[1][1]):
        #         # Simplify label to language abbreviation
        #         predict = l[0][0].replace("__label__", "")
        #         # If extracted language result is English
        #         if predict == correct:
        #             ok += 1
        #         else:
        #             print(f"Error at '{text}'")
        #             print(f"should be {correct}, predicted {predict} ")
        #             not_ok += 1

        #     print(f"ok = {ok}, not ok = {not_ok}")
    else:
        # Indicate document is in another language mostly and won't be adecuate to be used for k-means
        return False

# %% [markdown]
# ## Part of Speech(POS) tagging function to Find and store into DF POS via spacy

# %%
# Install spacy module by running through os library in terminal/console
# !python3 -m spacy download en_core_web_lg

# %%
# Create Spacy tokenizer based on downloaded language model in english
nlp = spacy.load("en_core_web_lg")

# Function to assign POS types into df
def word_types(doc):
    nouns = []
    adjectives = []
    verbs = []
    lemmas = []
    nav = []
    for token in doc:
        # Extract and append lemmas to lemmas list
        lemmas.append(token.lemma_)
        # adjectives (and adverbs)
        if token.pos_ == "ADJ": #or token.pos_ == "ADV":
            adjectives.append(token.lemma_)
            nav.append(token.lemma_)
        # nouns
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            nouns.append(token.lemma_)
            nav.append(token.lemma_)
        # verbs
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            verbs.append(token.lemma_)
            nav.append(token.lemma_)
            
    return (nouns, adjectives, verbs, nav, lemmas)

# %% [markdown]
# ## Main tokenization and tagging function

# %%
# Navigate through each row to set POS tags in corresponding locations
def pos_tagging(a:pd.DataFrame):
    # Iterate through rows to get text of each row
    for index, row in a.iterrows():
        # Check if language is considered English
        l = detect_language(row['body_text'])
        # If returns True because it is in English, perform tokenization
        if l: 
            # Use Spacy library to create tokens with text of document selected
            tokens = nlp(row['body_text'])
            # If produced tokens amount to greater than 30
            if len(tokens) >= 30:
                (nouns, adjectives, verbs, nav, lemmas) = word_types(tokens)
                a.at[index, 'nouns'] = "|".join(nouns)
                a.at[index, 'adjetives'] = "|".join(adjectives)
                a.at[index, 'verbs'] = "|".join(verbs)
                a.at[index, 'nav'] = "|".join(nav) # combinations
                a.at[index, 'lemmas'] = "|".join(lemmas)
                a.at[index, 'num_tokens'] = len(tokens)

# %%
# Temporary small df for testing
# a = df.copy(deep=True)
# a = a[:10]
# pos_tagging(a)
# a

# %%
# Function call for pos tagging extraction - averages 18 minutes of run
pos_tagging(df)

# %%
df.head(2)

# %% [markdown]
# ## If in need to start off from beginning but not perform clean

# %%
# df = pd.read_pickle('articles_cleaned.pkl')

# %% [markdown]
# ## Cleanup extended df from rows with no tags data and formatting issues

# %%
# Remove from df rows with empty cells in the POS tag columns
a = df.loc[~df['nouns'].isnull()]
df = a
# Reset index
df = df.reset_index(drop=True)

# %%
# Check if any null values still contained
df.loc[df['nouns'].isnull()]

# %%
# Output df
df.sample(3)

# %% [markdown]
# # Store and extract data as DB and Pickle files

# %%
# Store df also in a local sqlite3 database for easy management and create new pickle file with edited format
df.to_pickle('articles_extended.pkl')

# Transform list on tokens column into text due to sqlite3 requirements in data types
sql = sqlite3.connect('articles_extended.db')
df.to_sql('articles_extended', sql, if_exists='replace')
sql.close()

# %%
df = pd.read_pickle('articles_extended.pkl')

# %% [markdown]
# # Create document-term matrix using the TfidfVectorizer applied to the dataframe's column nouns

# %%
# Create a tf-idf vectorizer containing stopwords
tfidf_vectorizer = TfidfVectorizer(stop_words=list(stop_words), min_df=10, sublinear_tf=True, use_idf=True)
# Use fit transform on vectorizer with nouns data - standard deviation, then mean of data to scale values
tfidf_vectors = tfidf_vectorizer.fit_transform(df["nouns"])
# Create df from tfidf results and values
tfidf_df = pd.DataFrame(tfidf_vectors)

# %%
tfidf_df

# %% [markdown]
# # Extract nouns as word tokens from df['nouns'] column

# %%
# For word split by given specific regex, which has been set to all lowercase, and is not found in stop_words
# And is found in the text within each row of a df column called 'nouns'
gensim_words = [[w for w in re.split(r'[\\|\\#]', doc.lower()) if w not in stop_words]
                    for doc in df["nouns"]]
# Output list of nouns per row that has excluded stopwords
gensim_words

# %% [markdown]
# # Train Word2Vec model using df['nouns'] data to obtain word embeddings

# %%
'''
Use gemsin W2V model that implements skip-grams and continuous-bag-of-words
to capture conceptual similarities and train it using the word tokens from 
df to consider words that appear at least 5 times
'''
w2v = Word2Vec(gensim_words, min_count=5)
w2v.wv.save_word2vec_format("articles_extended.w2v")
# Get the keyedModel vectors
word_vectors = w2v.wv
# Example of content, aka number of keys or features, and their corresponding index
print(len(word_vectors.key_to_index.keys())) # word keys and length
word_vectors.key_to_index

# %% [markdown]
# # Represent each document via a one real-valued embedding vector/document embedding using the previously produced TF-IDF weights AND w2v embeddings

# %%
''' 
Use semantic transformation to create the real-value embedding
of a document by averaging all the embedding values of features
found in said document
'''

# List containing averaged word vectors from tf/idf, aka doc vectors
doc_v = []

# Get features from the tfidf vectorizer
fn = tfidf_vectorizer.get_feature_names_out()

# Iterate over all TF/IDF vectors, aka all documents/rows
for i in tqdm(range(tfidf_vectors.shape[0])): # adding tqdm to account for time it takes to process
    # Create array with size 100 (vector dimensions)from 
    # w2v to start averaged word vector of a current document to 0
    v = np.zeros(word_vectors.vector_size)

    ''' 
    Get the row and column index vals for tfidf vectors with nonzeros
    as this represents the words that actually occur in the i-th document
    ''' 
    rows, cols = tfidf_vectors[i].nonzero()

    # For index number in columns array
    for c in cols:
        
        # Extract feature from tfidf list of features
        feature = fn[c]
        '''
        If feature word found in this document is also in the word embeddings of w2v,
        multiply its tfidf value against vectors from embedding
        Example, decimal like 0.03849398273 * array['100 vectors values here']
        Then add it to v array of started as zeros of same length
        So basically, mutiply both vectors for same word from different methods,
        Then total sum all these vectors for all the words existing into a single vector
        to represent document
        '''
        if feature in word_vectors.key_to_index:
            wv = word_vectors[feature] # Extract embedding of w2v of given word
            v += tfidf_vectors[i][0,c] * (wv)
            
    ''' 
    Once you add all vectors from w2v * value of given features,
    normalize it if needed when the total values greater than 0
    '''
    if np.linalg.norm(v) > 0:
        v = v/np.linalg.norm(v)
    # Add normalized averaged vector value into the list
    doc_v.append(v)

# %% [markdown]
# ## Test embeddings provided by Word2Vec

# %%
# Test document embeddings given a keyword related to disaster topic
token = 'tornado'
# Get embedding/vector for said token from word vectors produced by word2Vec
token_v = word_vectors[token]
token_v # Array of 100 vals considered the dimensions of the word

# %% [markdown]
# ## Test document embeddings and its similarities to token word

# %%
'''
Test if vectors appropiate by calculating cosine similarity distance to find out the document
most related to this token
'''
doc_list = cosine_similarity(doc_v, [token_v]) # Must be a double-nested list
# Lenght is all docs available in document embeddings/df
print(len(doc_list))
# Extract most similar document index and output
the_doc = doc_list.argmax()
print('Article title: ', df.iloc[the_doc]['title'])
df.iloc[the_doc]

# %%
# Another test
token_v = word_vectors['hurricane']
doc_list = cosine_similarity(doc_v, [token_v]) # Must be a double-nested list
# Lenght is all docs available in document embeddings/df
print(len(doc_list))
# Extract most similar document index and output
the_doc = doc_list.argmax()
print('Article title: ', df.iloc[the_doc]['title'])
df.iloc[the_doc]

# %% [markdown]
# # Perform k-means clustering algorithm to group documents based on similarities which will be found via the document embeddings created through tf-idf and w2v

# %%
'''
k-means function to calculate the cluster given a:
    params:
        - numint => number of desired clusters
        - doc_v:list => produced document embeddings list
        - print:bool => indicate if to print clusters as process being performed
    returns:
        - dictionary
        - indirectly edits df provided
'''
def k_clusters(df:pd.DataFrame, num_clusters:int, doc_v:list, printit:bool):
    # Dictionary to store clusters
    clusters = {}
    ''' 
    KMeans from sklearn package call function to create clusters
    Params:     n_clusters = number of desired clusters
                random_state = control level of randomness
    '''
    kmeans_doc = KMeans(n_clusters=num_clusters, random_state=42).fit(doc_v)

    # Get sets and index of documents in df
    sets = list([(i,d) for d,i in enumerate(kmeans_doc.labels_)])
    # Sort based on cluster number from 0 to cluster_num
    sets.sort(key=lambda x: x[0])

    # Store in dictionary and print out docs based on cluster number
    for i in range(num_clusters):

        # Get index representing document number
        cd = [tuple[1] for tuple in sets if tuple[0] == i]

        # Extract indexes and titles and store in dictionary
        clusters[i] = list(zip(cd, df.iloc[cd]['title'].tolist()))

        # Extract indexes and titles for print output
        # df.iloc[cd]['cluster_num'] = i
        df.loc[cd,'cluster_num'] = i
        
        # Print list of article titles
        if printit:
            print('passed')
            titles = pd.DataFrame(df.iloc[cd]['title'], columns=['title'])
            titles = titles.rename_axis('Document Number')
            print(f'\nCluster {i} contains a total of {len(cd)} documents with the following titles:\n')
            print(titles)

    return kmeans_doc, clusters

# %%
# Create duplicate df and call K-Means algorithm function
df_temp = df.copy(deep=True)
num_c = 25
kmeans_c, cluster_dict = k_clusters(df_temp, num_c, doc_v, True)

# %%
# Output sample df with clusters set for 25
df_temp.sample(4)

# %%
# Store df as picke if needed for temporary analysis
df_temp.to_pickle(f'k_cluster_{num_c}.pkl')

# %% [markdown]
# # Cluster Mean Similarity to check levels of similarity between clusters created by program

# %%
def mean_similarity(kmeans_doc, num_clusters, doc_v):
    # Make documents embeddings into a dataframe
    dv_df = pd.DataFrame(doc_v)

    # Navigate through all clusters
    clusters_mean = []

    for cluster in range(num_clusters):
        # Get sets and index of documents in df
        sets = list([(i,d) for d,i in enumerate(kmeans_doc.labels_)])
        # Sort based on cluster number from 0 to cluster_num
        sets.sort(key=lambda x: x[0])

        # Get document index for all documents in said cluster
        doc_indx = [tuple[1] for tuple in sets if tuple[0] == cluster]
        cluster_df = dv_df.iloc[doc_indx]

        # Create cosine similarity matrix for all values in cluster
        pairs = cosine_similarity(cluster_df, cluster_df)

        # Extract non-repeated pairs in cosine matrix
        count = 1
        count_pairs = 0
        suma = 0
        for i in range(len(pairs)):
            for j in range(count, len(pairs[0])):
                count_pairs += 1 # Tracking # of pairs
                suma += pairs[i][j] # add up cosine value for pairs to total sum
                
            # Update count so to not repeat pairs on df
            count += 1
        # Divide the sum by number of pairs to calculate mean of clusters
        clusters_mean.append(suma/count_pairs)
        print(f'\nDocument cluster {cluster} has: \n{len(doc_indx)} Documents \n{count_pairs} document pairs \nMean similarity of {clusters_mean[cluster]}')
    
    return clusters_mean

# %%
means = mean_similarity(kmeans_c, num_c, doc_v)
means

# %%
df_temp

# %%
# looking at the clusters and their titles
clusters = {}
for index, i in df_temp.iterrows():
    if i['cluster_num'] not in clusters:
        clusters[i['cluster_num']] = {}
        clusters[i['cluster_num']]['titles'] = [i['title']]
        clusters[i['cluster_num']]['nouns'] = [i['nouns']]
    else:
        clusters[i['cluster_num']]['titles'].append(i['title'])
        clusters[i['cluster_num']]['nouns'].append(i['nouns'])

    

for i in clusters:
    print('Cluster: ', i)
    count = 0
    for j in clusters[i]['titles']:
        print(j)
        count += 1
        if count > 10:
            break
    print()


# %%
# creating a word cloud of the nouns for each cluster

# importing libraries
import wordcloud as wc
import matplotlib.pyplot as plt
# creating a word cloud for each cluster
for i in clusters:
    print('Cluster', i)
    current_nouns = []
    for j in clusters[i]['nouns']:
        current_nouns.append(j)
    # creating a word cloud
    current_nouns = ' '.join(current_nouns)
    wordcloud = wc.WordCloud(width = 800, height = 800,
                    background_color ='white',
                    min_font_size = 10).generate(current_nouns)
    # plotting the word cloud
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

    # saving the word cloud as a png file in 'word_clouds' folder
    os.makedirs('word_clouds', exist_ok=True)
    wordcloud.to_file('word_clouds/cluster_' + str(i) + '.png')


# %% [markdown]
# # END OF PART TWO


