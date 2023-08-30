import numpy as np

import math

from collections import defaultdict
import pandas as pd

from nltk.corpus import stopwords

from many_stop_words import get_stop_words

from nltk.stem import PorterStemmer #Stemmer
from textblob import Word #Lemmatize

from graphviz import Graph

import re
import nltk

from functools import reduce


from xploreapi import XPLORE

from sklearn.feature_extraction.text import CountVectorizer

import unittest

xploreID = '6g7w4kfgteeqvy2jur3ak9mn'
query = XPLORE(xploreID)
query.outputDataFormat='object'


# Calculate de absolute distance between the positions of two words in a document. word1_vector_pos and word2_vector_pos, represents the vectors with the word positions in the document. Using lambda functions.
def get_word_distances(word1_vector_pos, word2_vector_pos):
    
    # this empty list stores the output
    result = []
    
    # now, apply nested map function on both 
    # the created lists append the final output to
    # the "result" list
    list(map(lambda a: result.extend(map(a, word2_vector_pos)), map(lambda a: lambda b: abs(a-b), word1_vector_pos)))
    return result


# Get IEEE-Xplore article
def get_ieee_explore_article(parameter, value):
    query = XPLORE(xploreID)
    query.outputDataFormat='object'
    query.addParameter(parameter, value)
    data = query.callAPI()
    return data['articles'][0]['title'] + '. ' + data['articles'][0]['abstract']


# Get IEEE-Xplore Ranking
# q : query string
# nr_documents: number of retrieved documents results
def get_ieee_explore_ranking(q, max_records):
    query = XPLORE(xploreID)
    query.outputDataFormat='object'
    query.maximumResults(max_records)
    query.queryText(q)
    data = query.callAPI()
    results = data['articles']
    
    return results


# Transform the ranking array in one string document.
# if weighted <> none : the document will be weighted depending on its position in the ranking, 
# by multiplying its text (title + abstract) with the corresponding factor.
def get_ranking_as_string(results, weighted='none') :
    ranking = ''
    results_size = len(results)
    for index, article in enumerate(results):
        if (weighted=='linear'):
            factor = results_size - index
            ranking = ranking + ' ' + (article['title'] + '. ' + article['abstract'] + ' ') * factor
        elif (weighted=='inverse'):
            factor = math.ceil(results_size/(index+1))
            ranking = ranking + ' ' + (article['title'] + '. ' + article['abstract'] + ' ') * factor
        else:
            ranking = ranking + ' ' + article['title'] + '. ' + article['abstract']
    return ranking


def get_ranking_as_list(results, atribute_list):
    results_size = len(results)
    ranking_list = []
    for index, article in enumerate(results):
        ranking_string = ''
        for atribute in atribute_list:
            ranking_string = ranking_string + ' ' + article[atribute]
        ranking_list.append(ranking_string)
    return ranking_list



# Execute some text processing tasks
def text_transformations(parragraph, stop_words_list, lema=True, stem=True):   
    
    # Low the string
    parragraph = parragraph.lower()
    
    # Remove puntuation
    tokens = nltk.word_tokenize(parragraph)
    filtered_parragraph = [w for w in tokens if w.isalnum()]
    
    # Remove Stopwords
    if(len(stop_words_list)>0):
        filtered_parragraph = list(filter(lambda word_of_parragraph: (word_of_parragraph not in stop_words_list), filtered_parragraph))
    
    # Apply lematization
    if(lema):
        filtered_parragraph = list(map(lambda word_filtered_parragraph: Word(word_filtered_parragraph).lemmatize(), filtered_parragraph))
    
    # Stemmer
    if(stem):
        filtered_parragraph = list(map(lambda word: st.stem(word), filtered_parragraph))
    
    final_string = ' ' . join(map(str, filtered_parragraph))
    
    return final_string



# Calculate a matrix containing the terms positions from a group (list) of cocuments
def get_documents_positions_matrix(documents):
    term_positions_matrix = []
    for doc in documents:
        positions_dict = get_term_positions_dict(doc)
        term_positions_matrix.append(positions_dict)
    return term_positions_matrix


# Calculates a vecinity matrix from a list of documents
# - The @document_positions_matrix is an array of dictionaries
# - The result matrix is also an array of dictionaries 
# - Each dictionary corresponds to a a document or parragraph
# 
def get_vecinity_matrix(document_positions_matrix, reference_term, limit_distance, sumarize, include_reference_term):
    vecinity_matrix = []
    for doc_positions_dic in document_positions_matrix:
        document_term_vecinity_dict = get_document_term_vecinity_dict(doc_positions_dic, reference_term, limit_distance, sumarize, include_reference_term)
        vecinity_matrix.append(document_term_vecinity_dict)
    return vecinity_matrix


# Calculates the vecinity of a term in a doument
# where:
# document_positions_dict : is a dictionary with the positions of all terms in a document
# reference_term : is a term used as reference for calculating wich terms are in its vecinity
# limit_distance : is the maximal distance of terms used to calculate the vecinity
# sumarize : is used to define the function to sumarize the distance of the terms in the vecinity
def get_document_term_vecinity_dict(document_positions_dict, reference_term, limit_distance, sumarize='none', include_reference_term=True):
    
    # Create the empty dictionary
    vecinity_dict = {}
    
    # Get the term positions of the reference term
    reference_term_positions = document_positions_dict[reference_term]
    
    # Calculate all terms in document_positions_dict that are at distance limit_distance (or closer) to the reference_term
    # and return a list of these terms and their corresponding distances
    for term, term_positions in document_positions_dict.items():
        
        if((term != reference_term) or (include_reference_term)): # Evita que se compare el termino de referencia consigo mismo
            
            # Calculate the distance between the reference term and the rest of terms
            neighborhood_positions = calculate_term_positions_distances(reference_term_positions, term_positions, limit_distance)
            
            if(len(neighborhood_positions)>0):
                if (sumarize == 'mean'):
                    vecinity_dict[term] = np.mean(neighborhood_positions)
                elif (sumarize == 'median'): 
                    vecinity_dict[term] = np.median(neighborhood_positions)
                else: 
                    vecinity_dict[term] = neighborhood_positions
                
    return vecinity_dict



# Calculate the a dictionary with the document's term positions
# See corresponding UNITTEST
def get_term_positions_dict(document):
    vectorizer = CountVectorizer()
    vector = vectorizer.build_tokenizer()(document)
    document_positions_dic = defaultdict(list)
    for i in range(len(vector)):
        document_positions_dic[vector[i]].append(i)
    return document_positions_dic


# Merge two "generic" dictionaries 
# See corresponding UNITTEST
def merge_dictionaries(dict_A, dict_B):
    for i in dict_A.keys():
        if(i in dict_B.keys()):
            dict_B[i] = dict_B[i]+dict_A[i]
        else:
            dict_B[i] = dict_A[i]
    return dict_B


# Merge two "graph" dictionaries 
# See corresponding UNITTEST
def merge_graph_dictionaries(g1, g2):
    for i in g1.keys():
        if (i in g2.keys()):
            g2[i]['frequency'] = g2[i]['frequency'] + g1[i]['frequency']
            g2[i]['distances'] = g2[i]['distances'] + g1[i]['distances']
        else:
            g2[i] = {'frequency': g1[i]['frequency'], 'distances': g1[i]['distances']} 
    return g2


# Compare the positions vectors of two terms, and return the list of distances of the terms 
# that are inside limit_distance
# See corresponding UNITTEST
def calculate_term_positions_distances(term_positions1, term_positions2, limit_distance=float("inf")):
    term_distances = [] 
    for pos1 in term_positions1:
        for pos2 in term_positions2:
            absolute_distance = abs(pos1-pos2)
            if (absolute_distance <= limit_distance):
                term_distances.append(absolute_distance)
    return term_distances


            
# Calculates a vecinity dictionary from all documents, merging their vecinities
def get_unique_vecinity_dict(document_positions_matrix):
    product = reduce((lambda x, y: merge_dictionaries(x,y)), document_positions_matrix)
    return product


# Calculates a vecinity dictionary from an array of graph dictionaries, merging their vecinities
def get_unique_graph_dictionary(graph_dictionaries_array):
    product = reduce((lambda x, y: merge_graph_dictionaries(x,y)), graph_dictionaries_array)
    return product


# Normalize the values of a dictionary using a range
# The range should be (lower_bound, upper_bound)    
def normalize_dictionary_values(dictio, range):
    a = dictio[max(dictio, key=dictio.get)]
    c = dictio[min(dictio, key=dictio.get)]
    b = range[1]
    d = range[0]
    
    if((a - c)>0):
        m = (b - d) / (a - c)
    else:
        m = (b - d) # term frequency dictionary have only sigle words (frequency=1)
        
    dictio.update((k, (m*(dictio[k]-c)+d)) for k in dictio.keys())
    #dictio.update((k, (m*(dictio[k]-c)+d)) for k in dictio.keys())
    return dictio
    
    
# Calculates the graph of the most frecuently neighboors of a term
def getGraphViz(search_key, neighboors_df, node_size='2', node_color='green'):
    g = Graph('G', filename='graph_search.gv', engine='neato')
    g.attr('node', shape='circle', fontsize='10')
    counter = 0
    g.node('0', label=search_key, root='true', fixedsize='true', width=node_size, style='filled', fillcolor='azure3', fontcolor='black')
    for keyword, dic in neighboors_df.items() :
        counter = counter + 1
        p_with = str(dic['frequency'])
        g.node("'" +str(counter)+"'", keyword, fixedsize='true', width=node_size, penwidth=p_with, color=node_color)
        g.edge('0', "'" +str(counter)+"'", label=str(dic['distance']), len=str(dic['distance']))
        
    return g
