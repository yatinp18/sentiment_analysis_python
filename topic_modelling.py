import pandas as pd
import nltk, string
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import string


df=open('f1_new.pickle','rb')
df=pickle.load(df)

no_topics=10
no_top_words=10


def normalize(text):
     return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]



def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]




for i in df.itertuples():
    print('QUESTION 1')
    q1=getattr(i,'question1')



    table1= str.maketrans({key: None for key in string.punctuation})
    q1 = q1.translate(table1)


    if(len(q1) < 5):
        print('0.0 failed')
        continue


    q1=[q1]
    print('ID',getattr(i,'id'))
    count_vectorizer = CountVectorizer(max_df=1.0, min_df=0.5, max_features=10, stop_words=None)
    tfidf1= count_vectorizer.fit_transform(q1)

    lda1 = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf1)

    f1=count_vectorizer.get_feature_names()
    print('Q1=',str(f1).strip('[').strip(']'))
    f1=str(f1).strip('[').strip(']')

    print('QUESTION 2')

    q2=getattr(i,'question2')

    table = str.maketrans({key: None for key in string.punctuation})
    q2 = q2.translate(table)

    if(len(q2) < 5):
        print('0.0 failed')
        continue


    q2=[q2]


    tfidf2= count_vectorizer.fit_transform(q2)
    lda2 = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf2)

    feature_names=count_vectorizer.get_feature_names()
    f2=count_vectorizer.get_feature_names()
    print('Q2=',str(f2).strip('[').strip(']'))
    f2=str(f2).strip('[').strip(']')

    print(cosine_sim(f1,f2))
