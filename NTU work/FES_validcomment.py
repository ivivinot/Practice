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
savetarget_path = "C:/Users/IRU-RA2/Desktop/Feedback/2023/FES2023_Subsetv1_VC.xlsx"
writer = pd.ExcelWriter(savetarget_path, engine ="xlsxwriter")
df = pd.read_excel(excel_file_path)
ws = ["A01a","A02a","A03.10.01","A04.17.01","A05.13.01","A05.14.01","A05.15.01","A08.02a","B01a","B04c","B05d","B06c","B07c","B08c", "B09b","C02"]
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

    data={"UID":nonblankID,i:nonblankCom, "No Comment":None,"Not heard/used before":None, "It's already good":None,"Valid Comment":None}
    dictdf = pd.DataFrame(data)
    dict[i] = dictdf

#Designed to have more false positive in No Comment/ didn't heard or didn't know/already good as it is easier to identify than valid comment

no_comment_keyword = ["nil","'","-","/",".","'","na","no comment", "n/a", "did not attend","none"]
no_heard_keyword = ["heard"]
good_keyword = ["good","best", "great"]
for i in ws:
    nocomment=[]
    noheard=[]
    alreadygood=[]
    validcomment=[]
    df=dict[i]
    comment = df[i]
    for review in comment:
        if (any(keyword in review for keyword in no_comment_keyword) and len(review)<30) or len(review)<4:
            nocomment.append(1)
            noheard.append(" ")
            alreadygood.append(" ")
            validcomment.append(" ")
        elif any(keyword in review for keyword in no_heard_keyword):
            noheard.append(1)
            nocomment.append(" ")
            alreadygood.append(" ")
            validcomment.append(" ")
        elif any(keyword in review for keyword in good_keyword) and len(review)<80:
            alreadygood.append(1)
            nocomment.append(" ")
            noheard.append(" ")
            validcomment.append(" ")
        else:
            validcomment.append(1)
            nocomment.append(" ")
            noheard.append(" ")
            alreadygood.append(" ")
    df["No Comment"]=nocomment
    df["Not heard/used before"]=noheard
    df["It's already good"]=alreadygood
    df["Valid Comment"]=validcomment
    dictdf = pd.DataFrame(df)

    df.to_excel(writer,sheet_name=i)
writer.close()      

