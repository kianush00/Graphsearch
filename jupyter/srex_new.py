import numpy as np
import math
from collections import defaultdict
import pandas as pd
from nltk.stem import PorterStemmer as st #Stemmer
from textblob import Word #Lemmatize
from graphviz import Graph
import nltk
from functools import reduce
from xploreapi import XPLORE
from sklearn.feature_extraction.text import CountVectorizer
import copy

#import unittest

xploreID = '6g7w4kfgteeqvy2jur3ak9mn'
query = XPLORE(xploreID)
query.outputDataFormat='object'




def get_ieee_explore_article(
        parameter: str, 
        value: str
        ) -> str:
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


def get_ranking_as_weighted_documents(
        results: list[dict], 
        weighted: str = 'none'
        ) -> list[dict]:
    """
    Transform the ranking array in a list of strings associated with their weights.
    if weighted <> none : the document will be weighted depending on its position in the ranking

    Parameters
    ----------
    results : list[dict]
        Array of documents (articles)
    weighted : str
        Type of weighting to be applied (it can be: 'none', 'linear' or 'inverse')
    
    Returns
    -------
    ranking : list[dict]
        A list of dictionaries (documents), each dictionary contains the title and the abstract of an article as a 
        text (string), along with its weight (float)
    """
    ranking = []
    results_size = len(results)
    for index, article in enumerate(results):
        # Calculate the weight depending on the argument value and the position of the document in the ranking
        factor = calculate_factor(weighted, results_size, index)
        new_text = article['title'] + '. ' + article['abstract']
        ranking.append({'text': new_text, 'weight': factor})
    return ranking


def calculate_factor(
        weighted: str, 
        results_size: int, 
        index: int
        ) -> float:
    """
    Calculate the weight depending on the argument value and the position of the document in the ranking
    """
    if (weighted=='linear'):
        factor = float(results_size - index)
    elif (weighted=='inverse'):
        factor = float(math.ceil(results_size/(index+1)))
    else:
        factor = 1.0

    return factor
    


def get_sentences_list_from_documents(
        list_of_documents: list[dict]
        ) -> list[dict]:
    """
    Transform the ranking texts into a list of sentences (text splitted by dots).
    """
    for document in list_of_documents:
        document['text'] = document['text'].split('. ')
    return list_of_documents


def do_text_transformations_by_document(
    documents_list: list[dict],
    stop_words_list: list[str], 
    lema: bool = True, 
    stem: bool = True
    ) -> list[dict]:
    """
    Apply some text transformations to a list of documents (Remove stopwords, punctuation, stemming, lematization)
    """

    for document in documents_list:
        document["text"] = list(map(lambda sentence: text_transformations(sentence, stop_words_list, lema, stem), document["text"]))

    return documents_list


def do_text_transformations_by_string_in_list(
    string_list: list[str],
    stop_words_list: list[str], 
    lema: bool = True, 
    stem: bool = True
    ) -> list[str]:
    """
    Apply some text transformations to a list of strings (Remove stopwords, punctuation, stemming, lematization)
    """
    processed_string_list = []
    for term in string_list:
        processed_string_list.append(text_transformations(term, stop_words_list, lema, stem))

    return processed_string_list


def text_transformations(
        sentence: str, 
        stop_words_list: list[str], 
        lema: bool = True, 
        stem: bool = True
        ) -> str:
    """
    Apply some text transformations to a sentence (Remove stopwords, punctuation, stemming, lematization)

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


def delete_sentences_without_refterms(
    documents_list: list[dict],
    reference_terms: list[str], 
    ) -> list[dict]:
    """
    Delete sentences from a list of sentences by document that do not contain a reference term

    Parameters
    ----------
    documents_list: list[dict]
        List of dictionaries (articles), each one contains a list (array) of sentences
    reference_terms: list[str]
        List of reference terms that must be contained in the sentences
    
    Returns
    -------
    documents_list_with_refterms : list[dict]
        List of documents that exclusively contain a reference term in its list of sentences
    """
    for document in documents_list:
        sentences_with_refterms = []
        for sentence in document['text']:
            for term in reference_terms:
                if term in sentence:
                    sentences_with_refterms.append(sentence)
                    break
        document['text'] = sentences_with_refterms

    return documents_list


def get_documents_positions_matrix(
        documents_list: list[dict]
        ) -> list[dict]:
    """
    Calculate a matrix containing the terms positions from a group (list) of documents.

    Parameters
    ----------
    documents_list : list[dict]
        List of documents

    Returns
    -------
    documents_list_with_matrix_positions : list[dict]
        A list of dictionaries, each dictionary contains the terms positions of a sentence
    """
    for document in documents_list:
        term_positions_list = []
        for sentence in document['text']:
            positions_dict = get_term_positions_dict(sentence)
            term_positions_list.append(positions_dict)
        document['text'] = term_positions_list

    return documents_list


def get_term_positions_dict(
        sentence : str
        ) -> defaultdict[str, list[int]]:
    """
    Calculate the a dictionary with the sentence's term positions.
    
    Parameters
    ----------
    sentence : str
        The sentence to calculate its positions by term
    
    Returns
    -------
    sentence_positions_dict : defaultdict[str, list[int]]
        A dictionary with a list of positions for each term of the sentence
    """
    vectorizer = CountVectorizer()
    vector = vectorizer.build_tokenizer()(sentence)
    sentence_positions_dict = defaultdict(list)
    for i in range(len(vector)):
        sentence_positions_dict[vector[i]].append(i)
    return sentence_positions_dict


def get_vecinity_matrix(
        documents_list_with_matrix_positions: list[dict], 
        reference_terms: list[str], 
        limit_distance: int, 
        include_reference_term: bool
        ) -> list[dict]:
    """
    Calculate a vecinity matrix from a list of documents.

    Parameters
    ----------
    documents_list_with_matrix_positions : list[dict]
        List of dictionaries (documents), each dictionary contains the terms positions by sentence
    reference_terms: list[str]
        Terms used as reference for calculating wich terms are in its vecinity
    limit_distance : int
        Maximal distance of terms used to calculate the vecinity
    include_reference_term : bool
        If True, the reference term is included in the vecinity
    
    Returns
    -------
    documents_list_with_matrix_positions : list[dict]
        A list of dictionaries, each dictionary contains the terms in the vecinity of the reference term and their
        corresponding distances for each sentence
    """
    for document in documents_list_with_matrix_positions:
        vecinity_list = []
        for term_positions_defaultdict in document['text']:
            document_term_vecinity_dict = get_document_term_vecinity_dict(term_positions_defaultdict, reference_terms, limit_distance, include_reference_term)
            vecinity_list.append(document_term_vecinity_dict)
        document['text'] = vecinity_list

    return documents_list_with_matrix_positions


def get_document_term_vecinity_dict(
        term_positions_defaultdict: defaultdict[str, list[int]], 
        reference_terms: list[str], 
        limit_distance: int,
        include_reference_terms: bool = True
        ) -> dict[str, dict[str, list[int]]]:
    """
    Calculate the vecinity of a list of reference terms in a sentence, limited by a specified distance.
    
    Parameters
    ----------
    term_positions_defaultdict : defaultdict[str, list[int]]
        Dictionary with the positions of all terms in a sentence
    reference_terms : list[str]
        List of terms used as reference for calculating which terms are in its vecinity
    limit_distance : int
        Maximal distance of terms used to calculate the vecinity
    include_reference_terms : bool
        If True, the reference term is included in the vecinity
    
    Returns
    -------
    vecinity_dict : dict[str, dict[str, list[int]]]
        A dictionary with the terms in the vecinity of the reference term and their corresponding distances
    """

    vecinity_dict = {}  # Create the empty dictionary
    ref_term_positions_dict = get_ref_term_positions_dict(term_positions_defaultdict, reference_terms)
    
    # Calculate all terms in term_positions_defaultdict that are at distance limit_distance (or closer) to the reference_terms
    # and return a list of these terms and their corresponding distances
    for term, term_positions in term_positions_defaultdict.items():
        if((term not in ref_term_positions_dict.keys()) or (include_reference_terms)): # Avoid comparing the term of reference with itself
            # Calculate the distance between the reference term and the rest of terms
            first_one = True
            for ref_term, ref_positions in ref_term_positions_dict.items():
                freq_neighborhood_positions = calculate_frequency_term_positions_distances(ref_positions, term_positions, limit_distance)

                if (any(frq > 0 for frq in freq_neighborhood_positions)):
                    if (first_one):
                        vecinity_dict[term] = {}
                        first_one = False
                    vecinity_dict[term][ref_term] = freq_neighborhood_positions

    return vecinity_dict


def get_ref_term_positions_dict(
        term_positions_defaultdict: defaultdict[str, list[int]], 
        reference_terms: list[str]
        ) -> dict[str, list[int]]:
    """Returns a dictionary to store the list of positions for each reference term, along with its splitted terms"""
    ref_term_positions_dict = {}

    for ref_term in reference_terms:
        ref_term_words = ref_term.split(' ')
        if (len(ref_term_words) > 1) and (ref_term_words[0] in term_positions_defaultdict.keys()):    # If the reference term contains more than one word
            for splitted_ref_term in ref_term_words:   # Get the term positions of each splitted reference term
                ref_term_positions_dict[splitted_ref_term] = term_positions_defaultdict[splitted_ref_term]
            ref_term_positions_dict.update(format_ref_term_positions_dict(ref_term_positions_dict, ref_term_words))
        else:
            if ref_term in term_positions_defaultdict.keys():
                ref_term_positions_dict[ref_term] = term_positions_defaultdict[ref_term]    # Get the term positions of the reference term
    
    return ref_term_positions_dict


def format_ref_term_positions_dict(
        ref_term_positions_dict: dict[str, list[int]],
        reference_term_words: list[str]
        ) -> dict[str, list[int]]:
    """Format the reference term positions dictionary, so that it can be used in the vecinity matrix."""
    ref_term_positions_dict_splitted = defaultdict(list)

    #Omit keys that are not in reference_term_words
    for key, value in ref_term_positions_dict.items():
        if key in reference_term_words:
            ref_term_positions_dict_splitted[key] = value
    
    new_dict = get_new_dict_formatted_ref_term_positions(reference_term_words, ref_term_positions_dict_splitted)
    formatted_dict = get_new_dict_formatted_ref_term_positions(reference_term_words, new_dict)
    
    return formatted_dict


def get_new_dict_formatted_ref_term_positions(
        reference_terms: list[str],
        ref_term_positions_dict: dict[str, list[int]]
        ) -> dict[str, list[int]]:
    """Get a new formatted reference term positions dictionary"""
    new_formatted_dict = defaultdict(list)

    for index in range(len(reference_terms)-1):
        for number1 in ref_term_positions_dict[reference_terms[index]]:
            for number2 in ref_term_positions_dict[reference_terms[index+1]]:
                if ((number1 + 1) == number2):
                    if index == 0:
                        new_formatted_dict[reference_terms[index]].append(number1)
                    new_formatted_dict[reference_terms[index+1]].append(number2)
    
    return new_formatted_dict


def calculate_frequency_term_positions_distances(
        term1_positions: list[int], 
        term2_positions: list[int], 
        limit_distance: float = float("inf")
        ) -> list[int]:
    """
    Compare the positions vectors of two terms, and return the frequency quantity list 
    per distance between query terms and vecinity terms

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
    frequencies_per_distance : list[int]
        List of frequencies per distance between query terms and vecinity terms
    """
    frequencies_per_distance = [0] * limit_distance

    for term1_pos in term1_positions:
        for term2_pos in term2_positions:
            absolute_distance = abs(term1_pos-term2_pos)
            if (absolute_distance <= limit_distance):
                frequencies_per_distance[absolute_distance-1] += 1
    
    return frequencies_per_distance


def get_unique_vecinity_dict(
        document_positions_matrix: list[dict[str, list[int]]]
        ) -> dict[str, list[int]]:
    """
    Calculates a vecinity dictionary from all documents, merging their vecinities.

    Parameters
    ----------
    document_positions_matrix: list[dict[str, list[int]]]
        List of dictionaries, each dictionary contains the terms in the vecinity of the
        reference term and their corresponding distances

    Returns
    -------
    unique_vecinity_dict : dict[str, list[int]]
        A dictionary with all its vecinities merged from all documents
    """
    unique_vecinity_dict = reduce((lambda x, y: merge_dictionaries(x, y)), document_positions_matrix)
    return unique_vecinity_dict


def merge_dictionaries(dict_A: dict, dict_B: dict) -> dict:
    """Merge two generic dictionaries."""
    for key, value in dict_A.items():
        if key in dict_B.keys():
            dict_B[key] += value
        else:
            dict_B[key] = value
    return dict_B


def get_unique_graph_dictionary(graph_dictionaries_array):
    """Calculates a vecinity dictionary from an array of graph dictionaries, merging their vecinities."""
    product = reduce((lambda x, y: merge_graph_dictionaries(x,y)), graph_dictionaries_array)
    return product


def merge_graph_dictionaries(g1, g2):
    """Merge two "graph" dictionaries."""
    for i in g1.keys():
        if (i in g2.keys()):
            g2[i]['frequency'] = g2[i]['frequency'] + g1[i]['frequency']
            g2[i]['distance'] = g2[i]['distance'] + g1[i]['distance']
        else:
            g2[i] = {'frequency': g1[i]['frequency'], 'distance': g1[i]['distance']} 
    return g2


def normalize_dictionary_values(
        dictio: dict[str, int], 
        range: list
        ) -> dict[str, float]:
    """
    Normalize the values of a dictionary using a range.
    The range should be (lower_bound, upper_bound).
    This adjustment is used to scale the graph
    
    Parameters
    ----------
    dictio : dict[str, int]
        Dictionary to be normalized
    range : list
        Tuple with the lower and upper bounds of the range used to normalize the dictionary
    
    Returns
    -------
    dictio : dict[str, float]
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
