import numpy as np

import math

from collections import defaultdict
#import pandas as pd

#from nltk.corpus import stopwords

#from many_stop_words import get_stop_words
#from stop_words import get_stop_words

from nltk.stem import PorterStemmer as st #Stemmer
from textblob import Word #Lemmatize

from graphviz import Graph

#import re
import nltk

from functools import reduce


from xploreapi import XPLORE

from sklearn.feature_extraction.text import CountVectorizer

#import unittest

xploreID = '6g7w4kfgteeqvy2jur3ak9mn'
query = XPLORE(xploreID)
query.outputDataFormat='object'




def get_ieee_explore_article(parameter: str, value: str) -> str:
    """
    Get an article from IEEE-Xplore.
    
    Parameters
    ----------
    parameter
        Parameter used to search the article (e.g. 'article_number')
    value
        Value of the parameter used to search the article (e.g. '8600704')
    
    Returns
    -------
    article : str
        A string with the title and abstract of an article
    """
    query = XPLORE(xploreID)
    query.outputDataFormat='object'
    query.addParameter(parameter, value)
    data = query.callAPI()
    return data['articles'][0]['title'] + '. ' + data['articles'][0]['abstract']


def get_ieee_explore_ranking(
        query_text: str, 
        max_results: int
        ) -> list[dict]:
    """
    Get a ranking of articles from IEEE-Xplore.

    Parameters
    ----------
    query_text : str
        Text used to search the articles  (e.g. 'internet of things')
    max_results : int
        Maximum number of results to be returned

    Returns
    -------
    results : list[dict]
        A list of articles
    """
    query = XPLORE(xploreID)
    query.outputDataFormat='object'
    query.maximumResults(max_results)
    query.queryText(query_text)
    data = query.callAPI()
    results = data['articles']
    
    return results

# Comment from P.Galeas:
# This is a preliminary approach to assign some relevance to the documents in the ranking
# The first documents should be more relevant than the last documents in the ranking.
# This approach must be improved.
def get_ranking_as_string(
        results: list[dict], 
        weighted: str = 'none'
        ) -> str:
    """
    Transform the ranking array in one string document.
    if weighted <> none : the document will be weighted depending on its position in the ranking,
    by multiplying its text (title + abstract) with the corresponding factor.

    Parameters
    ----------
    results : list[dict]
        Array of documents (articles)
    weighted : str
        Type of weighting to be applied (it can be: 'none', 'linear' or 'inverse')
    
    Returns
    -------
    ranking : str
        A string with the ranking
    """
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



def text_transformations(
        sentence: str, 
        stop_words_list: list[str], 
        lema: bool = True, 
        stem: bool = True
        ) -> str:
    """
    Apply some text transformations to a sentence.

    Parameters
    ----------
    sentence : str
        String with the sentence to be transformed
    stop_words_list : list[str]
        List of stop words to be removed from the sentence
    lema : bool
        If True, lematization is applied
    stem : bool
        If True, stemming is applied
    
    Returns
    -------
    final_string : str
        The transformed sentence
    """
    
    # Low the string
    sentence = sentence.lower()
    
    # Remove puntuation
    tokens = nltk.word_tokenize(sentence)
    filtered_sentence = [tk for tk in tokens if tk.isalnum()]
    
    # Remove Stopwords
    if(len(stop_words_list)>0):
        filtered_sentence = list(filter(lambda word_of_sentence: (word_of_sentence not in stop_words_list), filtered_sentence))
    
    # Apply lematization
    if(lema):
        filtered_sentence = list(map(lambda word_filtered_sentence: Word(word_filtered_sentence).lemmatize(), filtered_sentence))
    
    # Stemmer
    if(stem):
        filtered_sentence = list(map(lambda word: st.stem(word), filtered_sentence))
    
    final_string = ' ' . join(map(str, filtered_sentence))
    
    return final_string



def get_documents_positions_matrix(documents: list[str]) -> list[dict[str, list[int]]]:
    """
    Calculate a matrix containing the terms positions from a group (list) of documents.

    Parameters
    ----------
    documents : list[str]
        List of documents

    Returns
    -------
    term_positions_matrix : list[dict[str, list[int]]]
        A list of dictionaries, each dictionary contains the terms positions of a document
    """
    term_positions_matrix = []
    for doc in documents:
        positions_dict = get_term_positions_dict(doc)
        term_positions_matrix.append(positions_dict)
    return term_positions_matrix


def get_vecinity_matrix(
        document_positions_matrix: list[dict[str, list[int]]], 
        reference_term: str, 
        limit_distance: int, 
        summarize: str, 
        include_reference_term: bool
        ) -> list[dict]:
    """
    Calculate a vecinity matrix from a list of documents.

    Parameters
    ----------
    document_positions_matrix : list[dict[str, list[int]]]
        List of dictionaries, each dictionary contains the terms positions of a document
    reference_term : str
        Term used as reference for calculating wich terms are in its vecinity
    limit_distance : int
        Maximal distance of terms used to calculate the vecinity
    summarize : str
        Used to define the function to summarize the distance of the terms in the vecinity
    include_reference_term : bool
        If True, the reference term is included in the vecinity
    
    Returns
    -------
    vecinity_matrix : list[dict]
        A list of dictionaries, each dictionary contains the terms in the vecinity of the reference term and their corresponding distances
    """
    vecinity_matrix = []
    for doc_positions_dic in document_positions_matrix:
        document_term_vecinity_dict = get_document_term_vecinity_dict(doc_positions_dic, reference_term, limit_distance, summarize, include_reference_term)
        vecinity_matrix.append(document_term_vecinity_dict)
    return vecinity_matrix


def get_document_term_vecinity_dict(
        document_positions_dict: dict[str, list[int]], 
        reference_term: str, 
        limit_distance: int, 
        summarize: str = 'none', 
        include_reference_term: bool = True
        ) -> dict:
    """
    Calculate the vecinity of a term in a document.
    
    Parameters
    ----------
    document_positions_dict : dict[str, list[int]]
        Dictionary with the positions of all terms in a document
    reference_term : str
        Term used as reference for calculating which terms are in its vecinity
    limit_distance : int
        Maximal distance of terms used to calculate the vecinity
    summarize : str
        Used to define the function to summarize the distance of the terms in the vecinity
        (it can be: 'mean', 'median' or 'none')
    include_reference_term : bool
        If True, the reference term is included in the vecinity
    
    Returns
    -------
    vecinity_dict : dict
        A dictionary with the terms in the vecinity of the reference term and their corresponding distances
    """

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
                if (summarize == 'mean'):
                    vecinity_dict[term] = np.mean(neighborhood_positions)
                elif (summarize == 'median'): 
                    vecinity_dict[term] = np.median(neighborhood_positions)
                else: 
                    vecinity_dict[term] = neighborhood_positions

    return vecinity_dict



# Calculate the a dictionary with the document's term positions
# See corresponding UNITTEST
def get_term_positions_dict(document : str) -> dict[str, list[int]]:
    """
    Calculate the a dictionary with the document's term positions.
    
    Parameters
    ----------
    document : str
        The document to calculate
    
    Returns
    -------
    document_positions_dic : dict[str, list[int]]
        A dictionary with a list of positions for each term of the document
    """
    vectorizer = CountVectorizer()
    vector = vectorizer.build_tokenizer()(document)
    document_positions_dic = defaultdict(list)
    for i in range(len(vector)):
        document_positions_dic[vector[i]].append(i)
    return document_positions_dic


# Merge two "generic" dictionaries 
# See corresponding UNITTEST
def merge_dictionaries(dict_A: dict, dict_B: dict):
    """Merge two "generic" dictionaries."""
    for i in dict_A.keys():
        if(i in dict_B.keys()):
            dict_B[i] = dict_B[i]+dict_A[i]
        else:
            dict_B[i] = dict_A[i]
    return dict_B


# Merge two "graph" dictionaries 
# See corresponding UNITTEST
def merge_graph_dictionaries(g1, g2):
    """Merge two "graph" dictionaries."""
    for i in g1.keys():
        if (i in g2.keys()):
            g2[i]['frequency'] = g2[i]['frequency'] + g1[i]['frequency']
            g2[i]['distances'] = g2[i]['distances'] + g1[i]['distances']
        else:
            g2[i] = {'frequency': g1[i]['frequency'], 'distances': g1[i]['distances']} 
    return g2


def calculate_term_positions_distances(
        term1_positions: list[int], 
        term2_positions: list[int], 
        limit_distance: float = float("inf")
        ) -> list[int]:
    """
    Compare the positions vectors of two terms, and return the list of 
    distances between the two terms that are inside limit_distance.

    Parameters
    ----------
    term1_positions : list[int]
        List of positions of the first term
    term2_positions : list[int]
        List of positions of the second term
    limit_distance : float
        Maximal distance of terms

    Returns
    -------
    term_distances : list[int]
        List of distances between the two terms that are inside limit_distance (Lenght = tp1_length * tp2_length)
    """
    term_distances = [] 
    for pos1 in term1_positions:
        for pos2 in term2_positions:
            absolute_distance = abs(pos1-pos2)
            if (absolute_distance <= limit_distance):
                term_distances.append(absolute_distance)
    return term_distances


            
def get_unique_vecinity_dict(document_positions_matrix):
    """Calculates a vecinity dictionary from all documents, merging their vecinities."""
    product = reduce((lambda x, y: merge_dictionaries(x,y)), document_positions_matrix)
    return product


def get_unique_graph_dictionary(graph_dictionaries_array):
    """Calculates a vecinity dictionary from an array of graph dictionaries, merging their vecinities."""
    product = reduce((lambda x, y: merge_graph_dictionaries(x,y)), graph_dictionaries_array)
    return product


def normalize_dictionary_values(
        dictio: dict, 
        range: list
        ) -> dict:
    """
    Normalize the values of a dictionary using a range.
    The range should be (lower_bound, upper_bound).
    This adjustment is used to scale the graph
    
    Parameters
    ----------
    dictio : dict
        Dictionary to be normalized
    range : list
        Tuple with the lower and upper bounds of the range used to normalize the dictionary
    
    Returns
    -------
    dictio : dict
        The normalized dictionary
    """
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
    
    
def getGraphViz(
        search_key: str, 
        neighboors_df: dict, 
        node_size: str = '2', 
        node_color: str = 'green'
        ) -> Graph:
    """Calculates the graph of the most frequently neighboors of a term."""
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
