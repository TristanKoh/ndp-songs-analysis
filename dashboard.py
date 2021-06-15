PATH = "data"

import streamlit as st
from streamlit_plotly_events import plotly_events

import pandas as pd
import numpy as np
import sqlite3
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

import datetime as dt


# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

###################################
### Prep Data for Visualisation ###
###################################
df_lyrics = pd.read_csv(PATH + "/lyrics.csv", encoding='utf-8')

# String cleaning - remove newline characters
df_lyrics["lyrics"] = df_lyrics["lyrics"].apply(lambda s : s.replace("\n", " "))

# Calculate number of words and average word length for each song
import re
df_lyrics["num_words"] = df_lyrics["lyrics"].apply(lambda s : len(re.findall(r'\w+', s)))

def avg_word_length(s) :
    num_words = len(re.findall(r'\w+', s))
    list_of_words = s.split()
    total_num_char = sum(len(i) for i in list_of_words)
    return(total_num_char / num_words)

df_lyrics["avg_word_len"] = df_lyrics["lyrics"].apply(lambda s : avg_word_length(s))

# Convert year to datetime object
df_lyrics["year"] = pd.to_datetime(df_lyrics["year"], format = "%Y")

# Combining year with name of song
df_lyrics["combined_name"] = df_lyrics["year"].astype(str).str.cat(df_lyrics["name"], sep = ", ")


# Tokenise lyrics
corpus = df_lyrics["lyrics"].tolist()

vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
dtm = vectorizer.fit_transform(corpus)
tf_array = np.asarray(dtm.sum(axis=0))
vocab_dict = vectorizer.vocabulary_
vocab_dict = dict(sorted(vocab_dict.items(), key = lambda x : x[1]))

# Convert dtm and tf to pandas df
df_dtm = pd.DataFrame.sparse.from_spmatrix(dtm)
df_dtm.columns = list(vocab_dict.keys())
df_dtm["year"] = df_lyrics["year"]

# Frequency of words
df_tf = pd.DataFrame(data = {"word" : list(vocab_dict.keys()), "freq" : tf_array[0, ]})
df_tf = df_tf.sort_values(by = "freq", ascending=False)

# Freq of top 30 words, averaged across 31 songs
df_tf["avg_freq"] = df_tf["freq"].apply(lambda x : x/31)


# Function that accepts a row of the dtm, sorts by count (descending), extracts the top 3 most frequent words, appends the freq and words to a new dataframe
def top_3_per_year(df) :
    indices = df.index.tolist()
    df_top_3 = pd.DataFrame(data = {"words" : [], "freq" : []})
    for i in range(max(indices)) :
        df_tmp = df.iloc[[i]]

        # Index everything except the last column since that is the year column in df_dtm
        df_tmp = pd.DataFrame(data = {"year" : [df_lyrics["year"][i].strftime("%Y")] * (len(df_tmp.columns) - 1) , "words" : df_tmp.columns.tolist()[:-1], "freq" : df_tmp.iloc[0, :-1] })

        df_tmp.reset_index(drop = True, inplace = True)
        df_tmp["year"] = pd.to_datetime(df_tmp["year"], format = "%Y")
        df_tmp["freq"] = df_tmp["freq"].astype(int)

        df_tmp = df_tmp.sort_values(by = "freq", ascending=False, axis = 0)
        df_top_3 = df_top_3.append(df_tmp.head(3), ignore_index=True)
    
    df_top_3["year"] = pd.to_datetime(df_top_3["year"], format = "%Y")

    return df_top_3

df_top_3 = top_3_per_year(df_dtm)


############################
### Sentiment Analysis #####
############################

# Sentiment analysis
# First need to preprocess the original string, lowercase, lemmatise, but don't remove stop words.
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import string

def pre_process_lyrics(lyrics, remove_stopwords = False) :
    processed_str = ""
    
    if remove_stopwords == True :
        lyrics = "".join(u for u in lyrics if u not in string.punctuation or stopwords.words('english'))

    lyrics = "".join(u for u in lyrics if u not in string.punctuation)
    lyrics_split = lyrics.split()
    lemmatizer = WordNetLemmatizer()

    for s in lyrics_split :
        s = s.lower()
        s = lemmatizer.lemmatize(s)
        processed_str = processed_str + " " + s
    
    return processed_str

df_lyrics_sentiment = df_lyrics.copy()
df_lyrics_sentiment["pre_process"] = df_lyrics_sentiment["lyrics"].apply(lambda x: pre_process_lyrics(x))

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

df_lyrics_sentiment["sentiment"] = df_lyrics_sentiment["pre_process"].apply(lambda x : sid.polarity_scores(x))
df_lyrics_sentiment["compound"] = df_lyrics_sentiment["sentiment"].apply(lambda x : x["compound"])
df_lyrics_sentiment["neg"] = df_lyrics_sentiment["sentiment"].apply(lambda x : x["neg"])
df_lyrics_sentiment["neu"] = df_lyrics_sentiment["sentiment"].apply(lambda x : x["neu"])
df_lyrics_sentiment["pos"] = df_lyrics_sentiment["sentiment"].apply(lambda x : x["pos"])


############################
######## Word Cloud ########
############################

single_string = []

for lyrics in df_lyrics_sentiment["pre_process"].tolist() :
    single_string.append(" ".join(c for c in lyrics.split() if c not in stopwords.words('english')))

single_string = "".join(single_string)
wordcloud = WordCloud(width = 1200, height = 600, min_font_size = 8, background_color = "white").generate(text = single_string)











############################
### Dashboard elements #####
############################
st.title("Singapore NDP Songs Lyrics Analysis (1985 - 2020)")

with st.beta_expander("Summary statistics"):
    col1, col2 = st.beta_columns(2)

    # Table that sorts lyrics by number of words
    with col1:
        st.markdown("*Songs with the greatest to least number of words*")
        st.dataframe(df_lyrics[["combined_name", "num_words"]].sort_values(by = ["num_words"], ascending = False))
    

    with col2:
        st.markdown("*Songs with the greatest to least by average word length*")
        st.dataframe(df_lyrics[["combined_name", "avg_word_len"]].sort_values(by = ["avg_word_len"], ascending = False))


    # Line plot by total number of words
    st.markdown("*Graph of total number of words vs year of song*")
    fig1 = px.line(df_lyrics, x = "combined_name", y = "num_words",
                    labels = {"combined_name" : "Year, Name of Song", 
                            "num_words" : "Total number of words in song"})

    st.plotly_chart(fig1, use_container_width=True)

    
    # Line plot by average word length
    st.markdown("*Graph of total number of words vs year of song*")
    fig2 = px.line(df_lyrics, x = "combined_name", y = "avg_word_len",
                    labels = {"combined_name" : "Year, Name of Song", 
                            "avg_word_len" : "Average word length of lyrics"})

    st.plotly_chart(fig2, use_container_width=True)


    # Table for total count of terms from largest to smallest frequency
    st.markdown("*Number of occurrences of words across all years*")
    st.dataframe(df_tf)

    
    col1, col2 = st.beta_columns(2)

    with col1:
        # Lineplot of freq of words vs word
        st.markdown("*Graph of 30 most frequent words across all years*")
        fig3 = px.line(df_tf.head(30), x = "word", y = "freq",
                                labels = {"word" : "Word", 
                                "freq" : "Frequency"})

        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Lineplot of average freq of words per song vs word
        st.markdown("*Graph of 30 most frequent words averaged per song across all years*")
        fig3 = px.line(df_tf.head(30), x = "word", y = "avg_freq",
                                labels = {"word" : "Word", 
                                "avg_freq" : "Frequency"})

        st.plotly_chart(fig3, use_container_width=True)


    # Barchart of top 3 words per song across all years
    st.markdown("*Bar graph of 3 most frequent words for each song*")
    fig4 = px.bar(df_top_3, x = "year", y = "freq", color = "words",
                    labels = {"year" : "Year", 
                            "freq" : "Frequency of word in song"})

    st.plotly_chart(fig4, use_container_width=True)


with st.beta_expander("Sentiment analysis (using NLTK's vader)"):
    col1, col2, col3 = st.beta_columns(3)

    with col1:
        # Most positive songs
        st.markdown("*Rating most positive songs (sorted by pos score)*")
        st.dataframe(df_lyrics_sentiment[["combined_name", "neg", "neu", "pos", "compound"]].sort_values(by = "compound", ascending=False))

    with col2: 
        # Most negative song
        st.dataframe(df_lyrics_sentiment[["combined_name", "neg", "neu", "pos", "compound"]].sort_values(by = "neg", ascending=False))

    with col3:
        # Overall most positive song
        st.dataframe(df_lyrics_sentiment[["combined_name", "neg", "neu", "pos", "compound"]].sort_values(by = "compound", ascending=False))
    
    # Linegraph of positive, negative and compound score metrics over time
    st.markdown("*Sentiment analysis metrics vs year*")
    fig5 = px.line(pd.melt(df_lyrics_sentiment[["year", "neg", "neu", "pos", "compound"]], ["year"]), 
                    x = "year", y = "value", color = "variable",
                    labels = {"year": "Year", "value" : "Score"})
    
    st.plotly_chart(fig5, use_container_width=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

with st.beta_expander("Word cloud (after stopwords removal)"):
    fig = plt.imshow(wordcloud)
    ax = plt.axis("off")
    plt.show()
    st.pyplot()