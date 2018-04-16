import pandas as pd
df = pd.read_csv("/home/socomo/Desktop/NLP/train.csv")

#df.drop(df.columns[[0,1,2]], axis=1, inplace=True)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 500)

#duplicate questions
q1_dup = df.loc[df['is_duplicate'] == 1, 'question1']
q2_dup = df.loc[df['is_duplicate'] == 1, 'question2']
q1_dup = q1_dup.to_frame()
q2_dup = q2_dup.to_frame()
pd.concat([q1_dup,q2_dup], axis=1)[:10]


#non-duplicate questions
q1_nondup = df.loc[df['is_duplicate'] == 0, 'question1']
q2_nondup = df.loc[df['is_duplicate'] == 0, 'question2']
q1_nondup = q1_nondup.to_frame()
q2_nondup = q2_nondup.to_frame()


pd.concat([q1_nondup,q2_nondup], axis=1)[:10]


def preprocessing_text(s):
    import re
    s = re.sub(r"[^A-Za-z0-9^,\*+-=]", " ",s)
    s = re.sub(r"(\d+)(k)", r"\g<1>000", s) #expand 'k' to '000' eg. 50k to 50000
    s = re.sub(r"\;"," ",s)
    s = re.sub(r"\:"," ",s)
    s = re.sub(r"\,"," ",s)
    s = re.sub(r"\."," ",s)
    s = re.sub(r"\<"," ",s)
    s = re.sub(r"\^"," ",s)
    s = re.sub(r"(\d+)(/)", "\g<1> divide ", s) #change number/number to number divide number (eg. 2/3 to 2 divide 3)
    s = re.sub(r"\/"," ",s) #replace the rest of / with white space
    s = re.sub(r"\+", " plus ", s)
    s = re.sub(r"\-", " minus ", s)
    s = re.sub(r"\*", " multiply ", s)
    s = re.sub(r"\=", "equal", s)
    s = re.sub(r"What's", "What is ", s)
    s = re.sub(r"what's", "what is ", s)
    s = re.sub(r"Who's", "Who is ", s)
    s = re.sub(r"who's", "who is ", s)
    s = re.sub(r"\'s", " ", s)
    s = re.sub(r"\'ve", " have ", s)
    s = re.sub(r"can't", "cannot ", s)
    s = re.sub(r"n't", " not ", s)
    s = re.sub(r"\'re", " are ", s)
    s = re.sub(r"\'d", " would ", s)
    s = re.sub(r"\'ll", " will ", s)
    s = re.sub(r"'m", " am ", s)
    s = re.sub(r"or not", " ", s)
    s = re.sub(r"What should I do to", "How can I", s)
    s = re.sub(r"How do I", "How can I", s)
    s = re.sub(r"How can you make", "What can make", s)
    s = re.sub(r"How do we", "How do I", s)
    s = re.sub(r"How do you", "How do I", s)
    s = re.sub(r"Is it possible", "Can we", s)
    s = re.sub(r"Why is", "Why", s)
    s = re.sub(r"Which are", "What are", s)
    s = re.sub(r"What are the reasons", "Why", s)
    s = re.sub(r"What are some tips", "tips", s)
    s = re.sub(r"What is the best way", "best way", s)
    s = re.sub(r"e-mail", "email", s)
    s = re.sub(r"e - mail", "email", s)
    s = re.sub(r"US", "America", s)
    s = re.sub(r"USA", "America", s)
    s = re.sub(r"us", "America", s)
    s = re.sub(r"usa", "America", s)
    s = re.sub(r"Chinese", "China", s)
    s = re.sub(r"india", "India", s)
    s = re.sub(r"\s{2,}", " ", s) #remove extra white space
    s = s.strip()
    return s


df2 = df.copy()
df2['question1'] = df2['question1'].astype(str)
df2['question1'] = df2['question1'].map(lambda x: preprocessing_text(x))
df2['question2'] = df2['question2'].astype(str)
df2['question2'] = df2['question2'].map(lambda x: preprocessing_text(x))


def remove_stopwords(string):
    word_list = [word.lower() for word in string.split()]
    from nltk.corpus import stopwords
    stopwords_list = list(stopwords.words("english"))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    return ' '.join(word_list)



df2['question1'] = df2['question1'].astype(str)
df2['q1_without_stopwords'] = df2['question1'].apply(lambda x: remove_stopwords(x))
df2['question2'] = df2['question2'].astype(str)
df2['q2_without_stopwords'] = df2['question2'].apply(lambda x: remove_stopwords(x))



def get_char_length_ratio(row):
    return len(row['question1'])/max(1,len(row['question2']))

df2['char_length_ratio'] = df2.apply(lambda row: get_char_length_ratio(row), axis=1)


def get_synonyms(word):
    from nltk.corpus import wordnet as wn
    synonyms = []
    if wn.synsets(word):
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))


def get_row_syn_set(row):
    import nltk
    syn_set = [nltk.word_tokenize(row)]
    for token in nltk.word_tokenize(row):
        if get_synonyms(token):
            syn_set.append(get_synonyms(token))
    return set([y for x in syn_set for y in x])



df2['q1_tokens_syn_set'] = df2['q1_without_stopwords'].map(lambda row: get_row_syn_set(row))
import nltk
df2['num_syn_words'] = df2.apply(lambda x:
                                 len(x['q1_tokens_syn_set'].intersection(set(nltk.word_tokenize(x['question2'])))), axis=1)



from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

no_topics = 100

tf_vectorizer = CountVectorizer(max_df=1.0, min_df=0.5, max_features=10, stop_words='english')

tf1 = tf_vectorizer.fit_transform(df2[['question1']])
tf2 = tf_vectorizer.fit_transform(df2[['question2']])

lda1 = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf1)
lda2 = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf2)


import pickle
feature_1_out=open("f1_new.pickle","wb")
pickle.dump(df2,feature_1_out)
