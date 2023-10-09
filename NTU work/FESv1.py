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

#sklearn
from wordcloud import WordCloud
from gensim import models
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel

nlp = spacy.load("en_core_web_sm")

excel_file_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1.xlsx"
savetarget_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_output.xlsx"
writer = pd.ExcelWriter(savetarget_path, engine ="xlsxwriter")
df = pd.read_excel(excel_file_path)
ws = ["A03.08.01","A09.02a","C02"]
UID = df.iloc[0:,0]
dict = {}
for i in ws: #filter out a dictionary with own column and UID
    A03 = df[i]
    nonblankID =[]
    nonblankCom=[]

    for ID,Com in zip(UID,A03):
        if not pd.isna(ID) and not pd.isna(Com):
            nonblankID.append(ID)
            nonblankCom.append(Com.lower())
#    no_comment = [non_english(comment) for comment in nonblankCom]
    data={"UID":nonblankID,i:nonblankCom, "No Comment":None,"Not heard/used before":None, "It's already good":None,"Valid Comment":None}
    dictdf = pd.DataFrame(data)
#    df['No Comment']=df['No Comment'].astype(int)
    dict[i] = dictdf
#    print(dictdf)
#    dict[i].to_excel(writer,sheet_name=i)
#writer.close()

no_comment_keyword = ["nil","'","-","/",".","'","na","no comment"]
no_heard_keyword = ["heard"]
good_keyword = ["good","best"]
for i in ws:
    nocomment=[]
    noheard=[]
    alreadygood=[]
    validcomment=[]
    df=dict[i]
    comment = df[i]
    for review in comment:
        if any(keyword in review for keyword in no_comment_keyword) and len(review)<4:
            nocomment.append(1)
            noheard.append(0)
            alreadygood.append(0)
            validcomment.append(0)
        elif any(keyword in review for keyword in no_heard_keyword):
            noheard.append(1)
            nocomment.append(0)
            alreadygood.append(0)
            validcomment.append(0)
        elif any(keyword in review for keyword in good_keyword):
            alreadygood.append(1)
            nocomment.append(0)
            noheard.append(0)
            validcomment.append(0)
        else:
            validcomment.append(1)
            nocomment.append(0)
            noheard.append(0)
            alreadygood.append(0)
#    data={"UID":nonblankID,i:nonblankCom, "No Comment":nocomment,"Not heard/used before":noheard, "It's already good":alreadygood,"Valid Comment":validcomment}
    df["No Comment"]=nocomment
    df["Not heard/used before"]=noheard
    df["It's already good"]=alreadygood
    df["Valid Comment"]=validcomment
    dictdf = pd.DataFrame(df)
#    print(dictdf)
#    dict[i]=data
    df.to_excel(writer,sheet_name=i)
writer.close()      
#print(dict)
'''
postoken =[]

for i in ws:
    filteredcomment =[]
    fc = [j for j, value in enumerate(dict[i]["Valid Comment"]) if value == 1]#fc represent row that has valid comment
#    print(fc)
    for j in fc:
        filteredcomment.append(dict[i][i][j]) #this loop is to pull out the row from the original comment with the row number
#    print(filteredcomment)
#    postoken = nlp(filteredcomment)
    filteredcomment = [str(x) for x in filteredcomment if pd.notna(x) and not isinstance(x, float) and 
        'Nil' not in str(x) and 'nil' not in str(x) and 'NIL' not in str(x)]
    splited = [re.split(r'[\.+\,+\!+\?+\+\_+\-"]',sent) for sent in filteredcomment]
#    print(splited)

    def lemmatization(texts, allowed_postags =['NOUN', 'ADJ', 'ADV','VERB']):#'NOUN', 'ADJ', 'VERB', 'ADV'
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            print(doc)
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out    
    postoken=lemmatization(splited, allowed_postags = ["'NOUN', 'ADJ', 'ADV','VERB'"])
#    print(postoken)

#fc = [i for i, value in enumerate(dict["Valid Comment"]) if value == 1]
#filteredcomment = [dict[nonblankCom][i] for i in fc]  
    
#print(dict)
'''

'''
    def non_english(text):
        text_lower = text.lower()
       
        non_english_pattern = re.compile(r'[^\x00-\x7F*\!nil\-\.]+')
        return bool(non_english_pattern.search(text_lower))
'''    
'''   
 dict[i]=dictdf
    dict[i].to_excel(savetarget_path,sheet_name="f{i}",index=False)
#print(dict)
#data_df = pd.DataFrame(dict)
#print(dict)
with pd.ExcelWriter(savetarget_path, engine ="xlsxwriter") as writer:
    for sheet_name, dict_df in dict["C02"]:
        dict_df.to_excel(writer, sheet_name=sheet_name,index=False)

    A03 = df[i]
    segm=pd.DataFrame(UID)
    segm[i]=A03
    dict[i] = segm
    
    blankUID = []
    blankCom = []
    for ID,Com in zip(dict[i].iloc[0:,0], dict[i].iloc[0:,1]):
        if ID!="" and Com!="":
            blankUID.append(ID)
            blankCom.append(Com)
#    print(dict)

#A09 = dict["A03.08.01"].iloc[0:,1]
#print(A09.iloc[0:,1])
#segm.to_excel(savetarget_path,sheet_name=i,index=False)
'''
#print(dict["A03.08.01"])
'''
#----------Base
#display maximum table
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
numtopics = 1
minwordfreq = 2

# Load spaCy language model
#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load('en_core_web_sm',disable = ['parser','ner'])


#open files for analysis
#excel_file_path = "C:/Users/IRU-RA2/Downloads/traintopic.xlsx"
excel_file_path = "C:/Users/IRU-RA2/Desktop/Feedback/2022/A09.02abackup.xlsx"
#df = pd.read_excel(excel_file_path,sheet_name = "Sheet1")
df = pd.read_excel(excel_file_path)
#selection of excel train_data
train_row   = 1
train_column= 1
#selection of excel train_target
target_row   = 0
target_column= 0
#---------Date cleaning
#covert to list
train_data = df.iloc[train_row:, train_column] 
train_target = df.iloc[target_row:, target_column] 
#removing blank and Nil
data = [str(x) for x in train_data if pd.notna(x) and not isinstance(x, float) and 
        'Nil' not in str(x) and 'nil' not in str(x) and 'NIL' not in str(x)]
datas = [re.split(r'[\.+\,+\!+\?+\"]',sent) for sent in data]
#print(train_target)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc=True removes punctuations
data_words = list(sent_to_words(datas))
#print(data_words)

stoplist = ['the', 'I','m','self','our','who','where','after','but','t','am','their','what',
            'me','are','as','about','on','over','once','myself','off','ma','having','because','whom','herself'
            ,'ourselves','d','very','a','i','than','by','how','few','all','now','these','which','was','be','why',
            'some','so','she','is','below','itself','above','ve','had','or','were','only','each','his',
            'he','it','doing','when','until','o','from','them','other','into','with','same','can','your',
            'to','my','here','its','themselves','you','ll','before','re','y','himself','there','own','did','between',
            'that','for','her','yourselves','do','the','in','they','yours','through','this','while','been','and','at',
            'just','nor','if','has','those','you','feel','felt','think','we','us','let','of','please','will'
            ,'an','would','could','whether','yet','make', 'bring','really', "quite","also","too","rather","come",
            "maybe","more","ntu"]
data_stop = [
    [word for word in sentence if word not in stoplist]
    for sentence in data_words]
#print(data_stop) #removing stopwords

def lemmatization(texts, allowed_postags =['NOUN', 'ADJ', 'VERB', 'ADV']):#'NOUN', 'ADJ', 'VERB', 'ADV'
    texts_out = []
    for sent in texts:
        doc=nlp(" ".join(sent))
#        print(doc)
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out
text_train_lemm = lemmatization(data_stop, allowed_postags=['NOUN','VERB','ADV','ADJ'])
print(text_train_lemm)

def preprocess(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.text.lower()]
    return tokens
tokens = [preprocess(text) for text in text_train_lemm]

docnames = ["Doc" + str(i) for i in range(len(text_train_lemm))] # assinging of docnames
orig = [data[i] for i in range(len(text_train_lemm))]

dictionary = corpora.Dictionary(tokens)
bow_corpus = [dictionary.doc2bow(doc) for doc in tokens] #or use tokens
tfidf = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf[bow_corpus]
#print(tfidf)
#print(tfidf_corpus)

lda_gensim = LdaModel(corpus=tfidf_corpus,
                      num_topics=numtopics,
                      id2word=dictionary,
                      alpha='auto',
                      eta='auto',
                      iterations = 100,
                      random_state=42)

def generate_word_cloud(lda_model, topic_num):
    words = dict(lda_model.show_topic(topic_num, topn=(len(dictionary))))  # Get the top 20 words for the topic
    wordcloud = WordCloud(width=1200, height=800, background_color='white',colormap="plasma", max_words=500).generate_from_frequencies(words)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='lanczos')
    plt.title(f'')
    plt.axis('off')
    plt.show()

num_topics = lda_gensim.num_topics
for topic_num in range(num_topics):
    generate_word_cloud(lda_gensim, topic_num)
    
#plt.savefig(f'topic_{topic_num + 1}_wordcloud.png', bbox_inches='tight')
#plt.close()  # Close the figure to release resources

topics=[{"interface","navigate","website user","intuitive","search","organize","organise","design","different","upgrade","ui","ux","user","friendly","system"},
        {"information","info","confuse","confusing","concise","transparent"},
        {"email","reply","send","reply email","letter","mail", "sm", "announce","notification","notify"},
        {"link","break","update","click","exist","load","long"},
        {"matriculation process","card","account"},
        {"application","process","procedure","submission","document","instruction","step","submit"},
        {"scholarship","financial","aid","fee"},
        {"offer","acceptance","admission","accept","registration","interview","matriculation","date","overseas"},
        {"orientation","freshman orientation","freshmen orientation"},
        {"engagement","bus","medical","modules"}]

Topicsname = ["Make user friendly interface, ease of navigation, upgrading of website",
              "Provision of information",
              "Email, sms, letter notifications",
              "Resolve/improve links issues",
              "Improve on matriculation matters",
              "Better application process/procedure",
              "Better information on scholarship & financial aspects",
              "Improve admission/Acceptance/Matriculation/Registration processes",
              "More information/alert on orientation",
              "Others"]

topic_dataframe = [pd.DataFrame(index=docnames, columns=[f"Topic {i+1}"], data=0) for i in range(len(topics))]
for i,item in enumerate(text_train_lemm):
    topic_found = False
    for j, topic_keywords in enumerate(topics):
        if any(keyword in item for keyword in topic_keywords):
            topic_dataframe[j].loc[docnames[i],f"Topic {j+1}"]=1
            topic_found = True
    if not topic_found:
        topic_dataframe[-1].loc[docnames[i],"Topic 10"]=1


for i, Topicsname in enumerate(Topicsname):
    topic_dataframe[i].columns = [Topicsname]
topic_matrix = pd.concat(topic_dataframe, axis=1)

topic_matrix["Original Docu"] = orig
num_rows = len(data)

#print(topic_matrix)

excelpath = "C:/Users/IRU-RA2/.spyder-py3/workingfolder/topicsorterv5.xlsx"
topic_matrix.to_excel(excelpath, index = True)


#lemmas into freq distribution 
fd = nltk.FreqDist(lemmas)
common_items = fd.most_common(100)
    
#for item,freq in common_items:
#    print(f"{item}:{freq}")

finder = nltk.collocations.BigramCollocationFinder.from_words(lemmas)
common_group = finder.ngram_fd.most_common(100)

for item,freq in common_group:
    print(f"{item}:{freq}")
print(common_group)
#print(cleanlist)
'''