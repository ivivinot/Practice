# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:52:48 2023

@author: iru-ra2
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, spacy, gensim
import pprint
import collections as Counter
from autocorrect import Speller

#sklearn
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from gensim import models
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

excel_file_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_VC.xlsx"
savetarget_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_NounCloud.xlsx"
writer = pd.ExcelWriter(savetarget_path, engine ="xlsxwriter")
PoS = ["NOUN", "ADJ","VERB", "DET", "AUX", "ADP", "PART", "PRON", "PROPN"]#"NOUN", "ADJ","VERB", "DET", "AUX", "ADP", "PART", "PRON", "PUNCT", "NUM"
ws = ["A01a","A02a","A03.10.01","A04.17.01","A05.13.01","A05.14.01","A05.15.01","A08.02a","B01a","B04c","B05d","B06c","B07c","B08c", "B09b","C02"]
#df = pd.read_excel(excel_file_path, sheet_name =ws)
stoplist = ['the', 'I','m','self','our','who','where','after','but','t','am','their','what',
            'me','are','as','about','on','over','once','myself','off','ma','having','because','whom','herself'
            ,'ourselves','d','very','a','i','than','by','how','few','all','now','these','which','was','be','why',
            'some','so','she','is','below','itself','above','ve','had','or','were','only','each','his',
            'he','it','doing','when','until','o','from','them','other','into','with','same','can','your',
            'to','my','here','its','themselves','you','ll','before','re','y','himself','there','own','did','between',
            'that','for','her','yourselves','do','the','in','they','yours','through','this','while','been','and','at',
            'just','nor','if','has','those','you','feel','felt','think','we','us','let','of','please','will'
            ,'an','would','could','whether','yet','make', 'have']
def excelreader (path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    return (df)
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc=True removes punctuations
def preprocess(text):
    doc = nlp("".join(text))
    tokens = [token.text for token in doc if token.is_alpha and token.text.lower() not in stoplist]
    return tokens        
def lemmatization(texts, allowed_postags =PoS):#"NOUN", "ADJ","VERB", "DET", "AUX", "ADP", "PART", "PRON", "PUNCT", "NUM"
    texts_out = []
    noun_lemma = []
    adj_lemma = []
    verb_lemma = []
    det_lemma = []
    aux_lemma = []
    adp_lemma = []
    part_lemma = []
    pron_lemma = []
    propn_lemma = []
    for sent in texts:
        spell = Speller(lang='en')
        corrected_sent = spell("".join(sent))
        doc = nlp("".join(corrected_sent))

        texts_out.append(" ".join([token.lemma_ for token in doc if token.pos_ in ["NOUN","ADJ","PROPN","VERB"]]))
        for token in doc:
            if token.pos_ in allowed_postags:
    # Append the lemma to the corresponding list based on POS
                if token.pos_ == "NOUN":
                    noun_lemma.append(token.lemma_)
                elif token.pos_ == "ADJ":
                    adj_lemma.append(token.lemma_)
                elif token.pos_ == "VERB":
                    verb_lemma.append(token.lemma_)
                elif token.pos_ == "DET":
                    det_lemma.append(token.lemma_)
                elif token.pos_ == "AUX":
                    aux_lemma.append(token.lemma_)
                elif token.pos_ == "ADP":
                    adp_lemma.append(token.lemma_)
                elif token.pos_ == "PART":
                    part_lemma.append(token.lemma_)
                elif token.pos_ == "PRON":
                    pron_lemma.append(token.lemma_)
                elif token.pos_ == "PROPN":
                    propn_lemma.append(token.lemma_)
                    
    return {"texts_out": texts_out, "NOUN": noun_lemma, "ADJ": adj_lemma, "VERB": verb_lemma, "DET":det_lemma, "AUX":aux_lemma,"ADP":adp_lemma,"PART":part_lemma,"PRON":pron_lemma, "PROPN":propn_lemma}    
vectorizer = CountVectorizer(analyzer = "word",
                             min_df= 1,
                             stop_words = "english",
                             token_pattern ='[a-zA-Z0-0]{1,}',
                             max_features = 50000)


dict = []
word_dict = {}
count_dict = {}
for i in ws: #filter out a dictionary with own column and UID
    current=[]
    df = excelreader(excel_file_path,i)
    filteredcomment =[]
    fc = [j for j, value in enumerate(df["Valid Comment"]) if value == 1]
    df = [df.loc[o,i] for o in fc]
    filteredcomment = [str(x) for x in df if pd.notna(x) and not isinstance(x, float) and 
        'Nil' not in str(x) and 'nil' not in str(x) and 'NIL' not in str(x)]
    splited = [re.split(r'[\.+\,+\!+\?"]',sent) for sent in filteredcomment]

    lemmatized= lemmatization(splited, allowed_postags=PoS)
    lemmatized_text = lemmatized["texts_out"]
    current = ["".join(lemmatizing) for lemmatizing in lemmatized_text]
    


    data_vectorized = vectorizer.fit_transform(current)
    feature_names = vectorizer.get_feature_names_out()
    PoS_list = []
    PoS_Dict = []
    for POS in PoS:

        PoS_list=(lemmatized[POS])

        PoS_list = ["".join(lemmatizing) for lemmatizing in PoS_list]

        word_counts = {}
        word_list = {}
        for word in PoS_list:
            if word in word_counts:
                word_counts[word]+=1 
            else:
                word_counts[word] = 1

        word_dict[POS] = word_counts.keys()
        word_dict[f"{POS}_Count"] = word_counts.values()

    word_dict.update(word_dict)
    df = pd.DataFrame.from_dict(word_dict,orient="index")
    t_df = df.transpose()
    t_df.to_excel(writer,sheet_name=i)
    
    tokens = [preprocess(text) for text in lemmatized_text]
#    print(tokens)
    dictionary = corpora.Dictionary(tokens)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokens] #or use tokens
 
    lda_gensim = LdaModel(corpus=bow_corpus,
                          num_topics=2,
                          id2word=dictionary,
                          alpha='auto',
                          eta='auto',
                          iterations = 100,
                          random_state=42)
    
    
    def generate_word_cloud(lda_model, topic_num, filename):
        topic_terms = lda_model.get_topic_terms(topic_num, topn=len(dictionary))
        mask = np.array(Image.open("C:/Users/IRU-RA2/Downloads/cloud.png").convert("RGB"))
        words = {dictionary[id]: weight for id, weight in topic_terms}
        wordcloud = WordCloud(width=1200, height=800, background_color='white', colormap="PuBu_r", mode = "RGBA", max_words=200, mask = mask).generate_from_frequencies(words)
        image_colors = ImageColorGenerator(mask)
        plt.figure(figsize=[12, 8])
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
        plt.title('')
        plt.axis('off')
        plt.savefig(filename,format="png")
        plt.close()
        plt.show()
    
    num_topics = lda_gensim.num_topics
#    print(num_topics)
    
    a = ""
    generate_word_cloud(lda_gensim, 0, f"{i}.png")
writer.close()
