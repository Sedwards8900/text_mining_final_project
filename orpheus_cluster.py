# Imports
# To run console commands for installation
import os
os.system('pip install -r requirements.txt') # change to pip3 if not your current command

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
# For vectorization and clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
# Wordcloud analysis and visualization
import wordcloud as wc
import matplotlib.pyplot as plt
from datetime import datetime
# to avoid warnings
import warnings
warnings.filterwarnings('ignore')

# Function to store data at this checkpoint after formatting body_text
def save_files(df, name):
    df.to_pickle(f'{name}.pkl')

    sql = sqlite3.connect(f'{name}.db')
    df.to_sql(name, sql, if_exists='replace')
    sql.close()

# *************
# Filtering documents from df
'''
Make and filter df function call for convenience of user to use application
All actions on blocks above being performed in one call

    params: none
    returns: df filtered
'''
def make_filter_df():
    # Extracting articles df
    df = pd.read_pickle('articles.pkl')
    
    # Remove articles that do not have any text - run only once to not cause error
    df = df.loc[df['body_text'] != '']
    
    # Remove any 403 error files - assignments done to avoid issues with slicing dataframe
    a = df.loc[(df['title'].str.contains('403')) & (df['url'].isnull())]
    mask = ~df.index.isin(a.index)
    b = df.loc[mask]
    
    a = b.loc[(b['title'].str.contains('Forbidden')) & (b['url'].isnull())]
    mask = ~b.index.isin(a.index)
    c = b.loc[mask]
    df = c

    # Reset index after performing first cleanup
    df = df.reset_index(drop=True)

    return df

# *************
# Clean up text data to ensure POS tokens significance can be extracted properly by spacy

'''
Function to remove any emojis in text
'''
def remove_emojis(text):
    # Get emoji codes from library demoji
    demoji.download_codes()
    
    dem = demoji.findall(text)
    for item in dem.keys():
        text = re.sub(item, '', text)
    return text


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

# *************
# Tokenization and POS tagging via Spacy


# Language detection function to ensure all tokens will be properly tagged based on the english language via fasttext
'''
Function that detects if a text contains a great quantity of tokens in another language,
if it does, it is removed from our sample, but if it is below the threshold, we do not remove
the words due to the fact that natural disasters often use name of locations in the U.S. that
are rooted in words from other languages.

    params:     doc => text string

    returns:    boolean => indicates if document passes language check
'''
def detect_language(lang_model, doc):

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


'''
Function to assign POS types into df
    params: text of document
    returns: 5 lists containing tokens
''' 
def word_types(doc):
    # Variables to store tokens
    nouns = []
    adjectives = []
    verbs = []
    lemmas = []
    nav = []
    
    for token in doc:
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

## Main tokenization and tagging function

# Navigate through each row to set POS tags in corresponding locations
def pos_tagging(a:pd.DataFrame):
    # Download from web the model for language detection from fasttext
    model_filename = "lid.176.ftz"
    r = requests.get(f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{model_filename}")
    open(model_filename, 'wb').write(r.content)
    # load language identification model 
    lang_model = fasttext.load_model(model_filename)

    # Install spacy module by running through os library in terminal/console
    os.system("python3 -m spacy download 'en_core_web_lg'")

    # Create Spacy tokenizer based on downloaded language model in english
    nlp = spacy.load("en_core_web_lg")

    # Iterate through rows to get text of each row
    for index, row in a.iterrows():
        # Check if language is considered English
        l = detect_language(lang_model, row['body_text'])
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


# ****************                
# Create document-term matrix using the TfidfVectorizer applied to the dataframe's column nouns

''' 
tfidf vectorizer
    params: df
    returns: tfidf vectors object
'''
def tfidf_vectorize(df):
    # Create a tf-idf vectorizer containing stopwords
    tfidf_vectorizer = TfidfVectorizer(stop_words=list(stop_words), min_df=10, sublinear_tf=True, use_idf=True)
    # Use fit transform on vectorizer with nouns data - standard deviation, then mean of data to scale values
    tfidf_vectors = tfidf_vectorizer.fit_transform(df["nouns"])
    
    return tfidf_vectorizer, tfidf_vectors

# ****************  
# Train Word2Vec model using df['nouns'] data to obtain word embeddings
def w2v_embedder(df, name):
    # For word split by given specific regex, which has been set to all lowercase, and is not found in stop_words
    # And is found in the text within each row of a df column called 'nouns'
    gensim_words = [[w for w in re.split(r'[\\|\\#]', doc.lower()) if w not in stop_words]
                        for doc in df["nouns"]]
    '''
    Use gemsin W2V model that implements skip-grams and continuous-bag-of-words
    to capture conceptual similarities and train it using the word tokens from 
    df to consider words that appear at least 5 times
    '''
    w2v = Word2Vec(gensim_words, min_count=5)
    w2v.wv.save_word2vec_format(f"{name}.w2v")
    # Get the keyedModel vectors
    return w2v.wv

# ****************
# Represent each document via a one real-valued embedding vector/document 
# embedding using the previously produced TF-IDF weights AND w2v embeddings


''' 
Document embedder function that uses semantic transformation to create the real-value embedding
of a document by averaging all the embedding values of features
found in said document

    params: tfidf_vectors
            w2v_vectors
    returns:
            document vectors
'''
def doc_embedder(tfidf_vectorizer, tfidf_vectors, w2v_vectors):
    # List containing averaged word vectors from tf/idf, aka doc vectors
    doc_v = []

    # Get features from the tfidf vectorizer
    fn = tfidf_vectorizer.get_feature_names_out()

    # Iterate over all TF/IDF vectors, aka all documents/rows
    for i in tqdm(range(tfidf_vectors.shape[0])): # adding tqdm to account for time it takes to process
        # Create array with size 100 (vector dimensions)from 
        # w2v to start averaged word vector of a current document to 0
        v = np.zeros(w2v_vectors.vector_size)

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
            if feature in w2v_vectors.key_to_index:
                wv = w2v_vectors[feature] # Extract embedding of w2v of given word
                v += tfidf_vectors[i][0,c] * (wv)
                
        ''' 
        Once you add all vectors from w2v * value of given features,
        normalize it if needed when the total values greater than 0
        '''
        if np.linalg.norm(v) > 0:
            v = v/np.linalg.norm(v)
        # Add normalized averaged vector value into the list
        doc_v.append(v)
    return doc_v


# *****************
# Perform k-means clustering algorithm to group documents based on 
# similarities which will be found via the document embeddings created 
# through tf-idf and w2v
'''
k-means function to calculate the cluster given a:
    params:
        - numint => number of desired clusters
        - doc_v:list => produced document embeddings list
        - print:bool => indicate if to print clusters as process being performed

    returns:
        - kmeans function object
        - dictionary containing a tuples with index and title of documents per cluster
        * indirectly edits df provided by creating column num_cluster and storing cluster number
'''
def k_clusters(df:pd.DataFrame, num_clusters:int, doc_v:list, printit:bool):
    # Dictionary to store clusters
    clusters_dict = {}
    ''' 
    KMeans from sklearn package call function to create clusters
    Params:     n_clusters = number of desired clusters
                random_state = control level of randomness

    returns:    clusters object
    '''
    kmeans_doc = KMeans(n_clusters=num_clusters, random_state=42).fit(doc_v)

    '''
    Get sets and index of documents in df, 
    .labels_returns an array of all documents in doc vectors indicating cluster number
    hence we need to enumerate them to maintain their location in the doc vectors
    and correctly find them in the df
    '''
    sets = list([(i,d) for d,i in enumerate(kmeans_doc.labels_)])
    # Sort based on cluster number from 0 to cluster_num
    sets.sort(key=lambda x: x[0])

    # Store in dictionary and print out docs based on cluster number
    for i in range(num_clusters):
        # Create cluster key in dictionary
        clusters_dict[i] = {}
        
        # Get index representing document number
        cd = [tuple[1] for tuple in sets if tuple[0] == i]

        # Extract indexes, titles, and nouns and store in dictionary
        # clusters[i] = list(zip(cd, df.iloc[cd]['title'].tolist()))
        clusters_dict[i]['indexes'] = cd
        clusters_dict[i]['titles'] = df.iloc[cd]['title'].tolist()
        clusters_dict[i]['nouns'] = df.iloc[cd]['nouns'].tolist()

        # Assign cluster number inside dataframe for each document
        df.loc[cd,'cluster_num'] = i
        
        # Print list of article titles if indicated
        if printit:
            titles = pd.DataFrame(df.iloc[cd]['title'], columns=['title'])
            titles = titles.rename_axis('Document Number')
            print(f'\nCluster {i} contains a total of {len(cd)} documents with the following titles:\n')
            print(titles)

    return kmeans_doc, clusters_dict


# ******************
# Cluster Mean Similarity to check levels of similarity between clusters created by program
'''
Function to calculate mean similarity of each cluster by adding up the cosine similarity of all document pairs in each cluster

    params: kmeans_doc => clusters produced by the k-means algorithm
            num_clusters => designated number of clusters, must match same number given during k-means function
            doc_v => all document vectors as lists of numpy arrays

    returns: 
        - list of all mean similarities
        - updated cluster dictionary
'''
def mean_similarity(kmeans_doc, cluster_dict, num_clusters, doc_v):
    # Make documents embeddings into a dataframe
    dv_df = pd.DataFrame(doc_v)

    # Navigate through all clusters
    for cluster in range(num_clusters):

        # Extract document from df with document index from dictionary
        cluster_df = dv_df.iloc[cluster_dict[cluster]['indexes']]

        # Create cosine similarity matrix for all values in cluster
        pairs = cosine_similarity(cluster_df, cluster_df)

        # Extract and add up cosine similarity value from non-repeated pairs in cosine matrix
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
        # clusters_mean.append(suma/count_pairs)
        cluster_dict[cluster]['mean_similarity'] = suma/count_pairs
        cluster_dict[cluster]['total_pairs'] = count_pairs

        # Print results
        print(f"\nDocument cluster {cluster} has: \n{len(cluster_dict[cluster]['indexes'])} Documents \n{count_pairs} document pairs \nMean similarity of {cluster_dict[cluster]['mean_similarity']}")
    
    # Return list containing all mean values per cluster
    return cluster_dict

# ******************
# Use nouns in dictionary to produce a wordcloud of nouns for each cluster

'''
Function to create a word cloud for each cluster

    params: clusters dictionary

    returns: NONE

    outputs: PNG files per each cluster containing wordclouds
'''

def clusters_to_cloud(clusters_dict:dict):
    # Save the wordcloud image as a png file within the 'word_clouds' folder
    current = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # Make directory folder
    os.makedirs(f'word_clouds/word_clouds_{num_c}_{str(current)}', exist_ok=True) # Path in case function running again

    # Navigate through each cluster in dictionary
    for key in clusters_dict:

        # Transform list of all nouns into a string with spaces in between
        current_nouns = clusters_dict[key]['nouns']
        current_nouns = ' '.join(current_nouns)

        # Use WordCloud function to display nouns
        wordcloud = wc.WordCloud(width = 800, height = 800,
                        background_color ='white',
                        min_font_size = 10).generate(current_nouns)
                
        print('Cluster', key)
        # Plot the word cloud usint matplotlib
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()

        # Save image
        wordcloud.to_file(f"word_clouds/word_clouds_{num_c}_{str(current)}/cluster_{key}.png")


# *****************
# Main call block for all functions to be ran based on selected criteria from user
def run_program(cluster_num):
    # Extract df from articles.pkl file and filter out unnecessary docs
    df = make_filter_df()
    print('make_filter done')

    # Clean text within column 'body_text' in dataframe df - takes approx. 2 to 3 mins
    df['body_text'] = [clean_text(text) for text in df['body_text']]
    print('cleaning done')

    # Save df as pickle after cleaning
    save_files(df, 'articles_cleaned')
    print('pickle 1 done')

    # Extract pickle to skip above for testing
    # df = pd.read_pickle('articles_cleaned.pkl')

    # Perform tokenization and tagging - averages 18 min run
    pos_tagging(df)
    print('tagging done')

    # Filtering again
    # Remove from df rows with empty cells in the POS tag columns - reassignment temporary due to slicing issues
    a = df.loc[~df['nouns'].isnull()]
    df = a
    # Reset index
    df = df.reset_index(drop=True)
    print('filter 2 done')

    # Store files during checkpoint
    save_files(df, 'articles_extended')
    print('pickle 2 done')

    # Extract pickle again to skip above for testing
    # df = pd.read_pickle('articles_cleaned.pkl')

    # Document term matrix TFIDF
    tfidf_v, tfidf_vectors = tfidf_vectorize(df)
    print('tfidf done')

    # w2v embedder
    word_vectors = w2v_embedder(df, 'articles_extended')
    print('w2v done')

    # Document embedder
    doc_vectors = doc_embedder(tfidf_v, tfidf_vectors, word_vectors)
    print('doc_v done')

    # Clustering
    kmeans_clusters, clusters_dict = k_clusters(df, cluster_num, doc_vectors, True)
    print('kmeans done')

    # Store df with clusters as pickle if needed for temporary analysis
    current = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    df.to_pickle(f'./clusters_pickles/k_cluster_{cluster_num}_{str(current)}.pkl')
    print('pickle 3 done')

    # Calculate mean similarity
    clusters_dict = mean_similarity(kmeans_clusters, clusters_dict, cluster_num, doc_vectors)
    print('mean similarity done')

    # Perform wordcloud for each cluster
    clusters_to_cloud(clusters_dict)
    print('wordcloud done')

    return df, clusters_dict, doc_vectors, tfidf_vectors, word_vectors


# RUN MAIN CALL FUNCTION
# Set desired cluster number - can be changed at desired amount
cluster_num = 25

# Run main call - total average time is 
final_df, cluster_dict, doc_vecs, tfidf_vecs, w2v_vecs = run_program(cluster_num)



# END OF PART TWO
# *****************

