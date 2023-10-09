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

#sklearn
from wordcloud import WordCloud
from gensim import models
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

excel_file_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_output.xlsx"
savetarget_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_output2.xlsx"
writer = pd.ExcelWriter(savetarget_path, engine ="xlsxwriter")
PoS = ["NOUN", "ADJ","VERB", "DET", "AUX", "ADP", "PART", "PRON", "PROPN"]#"NOUN", "ADJ","VERB", "DET", "AUX", "ADP", "PART", "PRON", "PUNCT", "NUM"
ws = ["A08a.07","C02"]
#df = pd.read_excel(excel_file_path, sheet_name =ws)
stoplist = ['the', 'I','m','self','our','who','where','after','but','t','am','their','what',
            'me','are','as','about','on','over','once','myself','off','ma','having','because','whom','herself'
            ,'ourselves','d','very','a','i','than','by','how','few','all','now','these','which','was','be','why',
            'some','so','she','is','below','itself','above','ve','had','or','were','only','each','his',
            'he','it','doing','when','until','o','from','them','other','into','with','same','can','your',
            'to','my','here','its','themselves','you','ll','before','re','y','himself','there','own','did','between',
            'that','for','her','yourselves','do','the','in','they','yours','through','this','while','been','and','at',
            'just','nor','if','has','those','you','feel','felt','think','we','us','let','of','please','will'
            ,'an','would','could','whether','yet','make']
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
        doc = nlp(" ".join(sent))
#        print(doc)
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


Topic1=[{"interface","navigate","website user","intuitive","search","organize","organise","design","different","upgrade","ui","ux","user","friendly","system"},
        {"information","info","confuse","confusing","concise","transparent"},
        {"email","reply","send","reply email","letter","mail", "sm", "announce","notification","notify"},
        {"link","break","update","click","exist","load","long"},
        {"matriculation process","card","account"},
        {"application","process","procedure","submission","document","instruction","step","submit"},
        {"scholarship","financial","aid","fee"},
        {"offer","acceptance","admission","accept","registration","interview","matriculation","date","overseas"},
        {"orientation","freshman","freshmen"},
        {"engagement","bus","medical","modules"}]

Topicsname1 = ["Make user friendly interface, ease of navigation, upgrading of website",
              "Provision of information",
              "Email, sms, letter notifications",
              "Resolve/improve links issues",
              "Improve on matriculation matters",
              "Better application process/procedure",
              "Better information on scholarship & financial aspects",
              "Improve admission/Acceptance/Matriculation/Registration processes",
              "More information/alert on orientation",
              "Others"]

Topic2=[{"activity", "program","event","games","boring","crazy","hype"},
        {"information","info","confuse","confusing","concise","transparent"},
        {"email","reply","send","reply email","letter","mail", "sm", "announce","notification","notify"},
        {"link","break","update","click","exist","load","long"},
        {"matriculation process","card","account"},
        {"application","process","procedure","submission","document","instruction","step","submit"},
        {"scholarship","financial","aid","fee"},
        {"offer","acceptance","admission","accept","registration","interview","matriculation","date","overseas"},
        {"orientation","freshman","freshmen"},
        {"engagement","bus","medical","modules"}]

Topicsname2 = ["More or better activities, programmes, events, games",
              "More information, explanation, sharing",
              "More bonding/interaction/inclusivenss",
              "Better time management(e.g. less waiting time, back to back camp",
              "More interaction with seniors and professors",
              "Provide more academic information (Printed etc.)",
              "Longer duraction",
              "Shorter Duration",
              "Fees/Food/Goodies/Free Gifts",
              "Improve NTU website apps",
              "Campus/School tours",
              "More publicity",
              "More appropriate activities/Cheers",
              "Improve on online/virtual events. Want more virtual tours/events",
              "Want more physical/interactive events",
              "Covid-19 related",
              "Better date/Longer registration period",
              "Others"]

#dict = []
new_df = pd.DataFrame()
dict = {}
wstopic = {}

for no, i in enumerate(ws): #filter out a dictionary with own column and UID
    dict[i]={}
    list_name = f"topicname{no}"

    current=[]
    df = excelreader(excel_file_path,i)
    print(list_name)
    filteredcomment =[]
    vc = [j for j, value in enumerate(df["Valid Comment"]) if value == 1]

#    uid = [j for j, value in enumerate(df["Valid Comment"]) if value == 1]
    UID = [df.loc[o,"UID"] for o in vc]

    VC = [df.loc[o,i] for o in vc]
#    print(VC)
    dict[i]["UID"]=UID
    dict[i]["Valid Comment"]= VC
    lemm_vc = lemmatization(VC)["texts_out"]
    new_df = pd.DataFrame(dict[i])
#    print(pd.DataFrame(lemm_vc))
    #print(df)



#    topic_dataframe = [pd.DataFrame(index = dict[i], columns = [f"Topic {x+1}"],data=[[0]*len(dict[i])]) for x in (range(len(Topic))]
#[pd.DataFrame(index=docnames, columns=[f"Topic {i+1}"], data=0) for i in range(len(topics))]
#topic_patterns = {topic: re.compile(re.escape(topic), flags=re.IGNORECASE) for topic in topics}
    topic_dataframe = pd.DataFrame(columns=Topicsname1)
    for y,item in enumerate(VC):
        topic_found = False
        for j, topic_keywords in enumerate(Topic):
            #topic_patterns = {topic: re.compile(re.escape(topic), flags=re.IGNORECASE) for topic in topics}
            if any(keyword in item for keyword in topic_keywords):
                topic_dataframe.loc[y,Topicsname1[j]]=1
                topic_found = True
        if not topic_found:
            topic_dataframe.loc[y,"Others"]=1
    print(i)

    new_df = pd.concat([new_df,topic_dataframe], join="inner", axis=1)
    new_df.to_excel(writer,sheet_name=i)
writer.close()
''' 
    for i, Topicsname in enumerate(Topicsname):
        topic_dataframe[i].columns = [Topicsname]
    topic_matrix = pd.concat(topic_dataframe, axis=1)
    
    topic_matrix["Original Docu"] = orig
    num_rows = len(data)
    
    
    topic_matrix.to_excel(savetarge_path, index = True)
'''
