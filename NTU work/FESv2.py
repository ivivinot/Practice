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
from wordcloud import WordCloud
from gensim import models
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

excel_file_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_output.xlsx"
savetarget_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_output1.xlsx"
writer = pd.ExcelWriter(savetarget_path, engine ="xlsxwriter")
PoS = ["NOUN", "ADJ","VERB", "DET", "AUX", "ADP", "PART", "PRON", "PROPN"]#"NOUN", "ADJ","VERB", "DET", "AUX", "ADP", "PART", "PRON", "PUNCT", "NUM"
ws = ["A03.08.01","A09.02a","C02"]
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
        spell = Speller(lang='en')
        corrected_sent = spell("".join(sent))
        doc = nlp("".join(corrected_sent))
        print(doc)
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
        words = {dictionary[id]: weight for id, weight in topic_terms}
        wordcloud = WordCloud(width=1200, height=800, background_color='white', colormap="plasma", max_words=200).generate_from_frequencies(words)
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='lanczos')
        plt.title('')
        plt.axis('off')
        plt.savefig(filename,format="png")
        plt.close()
        plt.show()
    
    num_topics = lda_gensim.num_topics
#    print(num_topics)
    
    
    generate_word_cloud(lda_gensim, 0, f"{i}.png")
writer.close()

'''
        word_dict[POS] = list(word_counts.keys())
        word_dict[f"{POS}_Count"]= list(word_counts.values())
    dict.append(word_dict)
    word_frequency = pd.DataFrame(dict)
    word_frequency.to_excel(writer,sheet_name=i)
    
writer.close()
'''
'''        
#        print(word_list)
        count_dict[POS] =list(word_counts.values())
        #print(word_list)
    PoS_Dict["word"] = word_dict
    PoS_Dict["count"]= count_dict
    word_frequency = pd.DataFrame(PoS_Dict)
    t_word_freq = word_frequency.transpose()
    t_word_freq.to_excel(writer,sheet_name=i)
writer.close()
'''
#    dict[ws]=dict
#        word_counts_df = df.append(word_counts_df,ignore_index=True)


#        for word, count in word_counts.items():
#            print(f"{word}, {count}")
#        word_counts = Counter(PoS_list)
#        for word, count in word_counts.item():
#            print(f"{word}:{count}")
#        PoS_vect = vectorizer.fit_transform(PoS_list)
#        PoS_names = vectorizer.get_feature_names_out()
#        print(PoS_names)
#        PoS_Dict[POS] = PoS_vect

#        PoS_list = [" ".join(Poslist) for Poslist in PoS_list]

#   noun_list = lemmatized["noun"]

#    noun_vect = vectorizer.fit_transform(noun_list)
#    noun_feature = vectorizer.get_feature_names_out()
#    print(data_vectorized)
#    print(noun_vect)
#    print(noun_feature)
#    names = nlp(" ".join(current))
#    a=[token.lemma_ for token in names if token.pos_=="NOUN" and "ADJ" and "VERB" and "DET" and "AUX" and "ADP" and "PART" and "PRON"]
#    print(feature_names)
#    print(names)
#    print(a)
'''
    word_frequencies = data_vectorized.sum(axis=0)
    word_frequency_df = pd.DataFrame({'Word':feature_names,'Frequency':word_frequencies.flat})
#    word_frequency_df['POS']=names
    word_frequency_df = word_frequency_df.sort_values(by='Frequency',ascending=False)
    dict[i]=word_frequency_df
#    print(word_frequency_df)
#    print(word_frequencies)
    word_frequency_df.to_excel(writer,sheet_name=i)
'''
#print(dict)
#writer.close()      
#print(dict)
'''
    word_frequency = {word: 0 for word in a}
    for feature_name in feature_names:
        if feature_name in word_frequency:
            word_frequency[feature_name] += 1
        else:
            word_frequency[feature_name] = 1
        
    word_frequency_df =pd.DataFrame(word_frequency.item(),columns=["Word","Frequency"] )
    word_frequency_df = word_frequency_df.sort_values(by="Frequency", ascending = False)
    print(word_frequency_df)
'''    
'''
    print(a)
    word_frequencies = data_vectorized.sum(axis=0)
    word_frequency_df = pd.DataFrame({'Word':a,'Frequency':word_frequencies.flat})
    word_frequency_df = word_frequency_df.sort_values(by='Frequency',ascending=False)
    print(word_frequency_df)
'''

    

#    print(names)
#    print("Nouns:",[token.lemma_ for token in names if token.pos_=="NOUN"])

#    topic_freq = Counter(names.token.pos_=="NOUN" for topic in names)
#    words_frequencies = data_vectorized.toarray()
#    print(words_frequencies)
#    print(data_vectorized)
#    print(feature_names)
#    print("Nouns:",[token.lemma_ for token in names if token.pos_=="NOUN"])
#    noun_df =pd.DataFrame()
#    verb_df =pd.DataFrame()
#    adj_df  =pd.DataFrame()
#    noun_df = noun_df.append(words_frequencies[["NOUN"]])
#    noun_counts = noun_df.sum()
#    word_frequencies_df = pd.DataFrame({"NOUN": noun_counts})
    
#    print(word_frequencies_df)

#    dict[i] = current
#print(dict)


#        doc = nlp(" ".join(sent))
#        text_out.append(" ".join([token.lemma_ for token in doc]))
#        print(text_out)
#        for tok in text_out:
#            print(f"{tok.text:<10} {tok.tag_:<10} {tok.pos_:<10}")
#        doc = nlp("".join(token))
#        text_out.append(" ".join([token.lemma_ for token in doc if token.pos_ in "NOUN" and "VERB"]))
#       print(text_out)
        
#    tokeniser = list(sent_to_words(splited))
#    data_stop = [[word for word in sentence if word not in stoplist]for sentence in tokeniser]
#    print(data_stop)
#    postoken=lemmatization(data_stop, allowed_postags = ["'NOUN', 'ADJ', 'ADV','VERB'"])
#    print(postoken)
'''
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
    print(df)
















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



    def non_english(text):
        text_lower = text.lower()
       
        non_english_pattern = re.compile(r'[^\x00-\x7F*\!nil\-\.]+')
        return bool(non_english_pattern.search(text_lower))

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

