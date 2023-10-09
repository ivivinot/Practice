# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:33:40 2023

@author: iru-ra2
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

pd.set_option("display.max_column", None)
pd.set_option("display.max_rows",None)

# Sample documents
documents = [
    "I love ice cream.",
    "I love chocolate.",
    "Chocolate is an ingredient in ice cream.",
    "Hate Ice cream",
    "Hate chocolate",
    "hate chocolate ice cream"
]


# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert the matrix to a DataFrame for better visualization
df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(df)

def improved_classify_document(tfidf_df, threshold=0.5):
    classifications = []
    for index, row in tfidf_df.iterrows():
        ice_cream_score = row.get('ice', 0) + row.get('cream', 0)
        chocolate_score = row.get('chocolate', 0)
        score_difference = abs(ice_cream_score - chocolate_score)
        if score_difference < threshold:
            classifications.append('Both')
        elif ice_cream_score > chocolate_score:
            classifications.append('Ice cream')
        else:
            classifications.append('Chocolate')
    return classifications

improved_classifications = improved_classify_document(df)
print(improved_classifications)