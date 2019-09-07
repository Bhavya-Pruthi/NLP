# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:13:03 2019

@author: bhavy
"""

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

def convert_tag(tag):    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None
def doc_to_synsets(doc): 

    tokens = nltk.word_tokenize(doc)
    pos = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in pos]
    wntag = [convert_tag(tag) for tag in tags]
    ans = list(zip(tokens,wntag))
    sets = [ wn.synsets(x,y) for x,y in ans]
    final = [val[0] for val in sets if len(val) > 0]
    
    return final
def similarity_score(s1, s2):
    s=[]
    for i1 in s1:
        r=[]
        scores=[x for x in [i1.path_similarity(i2) for i2 in s2]if x is not None]
        if scores:
            s.append(max(scores))
    # Your Code Here
    
    return sum(s)/len(s)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)

paraphrases = pd.read_csv('paraphrases.csv')
def label_accuracy():
    from sklearn.metrics import accuracy_score
    df = paraphrases.apply(update_sim_score, axis=1)
    score = accuracy_score(df['Quality'].tolist(), df['paraphrase'].tolist())
    
#    return score

def update_sim_score(row):
    row['similarity_score'] = document_path_similarity(row['D1'], row['D2'])
    row['paraphrase'] = 1 if row['similarity_score'] > 0.75 else 0
    return row


