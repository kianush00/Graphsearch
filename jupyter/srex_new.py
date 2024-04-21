import numpy as np
import math
from collections import defaultdict
#import pandas as pd
from nltk.stem import PorterStemmer as st #Stemmer
from textblob import Word #Lemmatize
from graphviz import Graph
import nltk
from functools import reduce
from xploreapi import XPLORE
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from numpy import dot
#import copy

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
    Calculate a weight factor depending on the argument value ('linear' or 'inverse'), and 
    the position of the document in the ranking.

    Parameters
    ----------
    weighted : str
        The type of weighting to be applied. Can be 'linear', 'inverse', or any other value.
    results_size : int
        The total number of results/documents.
    index : int
        The position of the document in the ranking.

    Returns
    -------
    factor : float
        The calculated weight factor.
    """
    if (weighted=='linear'):
        factor = float((results_size - (index * 0.5)) / results_size)
    elif (weighted=='inverse'):
        factor = float(math.ceil(results_size / ((index * 0.5) + 1)) / results_size)
    else:
        factor = 1.0

    return factor
    


def get_sentences_list_from_documents(
        list_of_documents: list[dict]
        ) -> list[dict]:
    """
    Transform the text of each document in the input list into a list of sentences. Split the text by dots.

    Parameters
    ----------
    list_of_documents : list[dict]
        A list of dictionaries representing documents, where each dictionary has a 'text' key containing the document text.

    Returns
    -------
    list_of_documents_with_sentences_list : list[dict]
        A list of dictionaries representing documents, where each dictionary has a 'text' key containing a list of sentences.
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
    Apply some text transformations to a list of documents. Remove stopwords, punctuation, stemming, lematization

    Parameters
    ----------
    string_list : list[str]
        A list of documents to be transformed.
    stop_words_list : list[str]
        List of stop words to be removed from the sentence
    lema : bool
        If True, lematization is applied
    stem : bool
        If True, stemming is applied

    Returns
    -------
    transformed_documents_list : list[dict]
        A list of transformed documents.
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
    Apply text transformations to a list of strings. Remove stopwords, punctuation, stemming, lematization.

    Parameters
    ----------
    string_list : list[str]
        A list of strings to be transformed.
    stop_words_list : list[str]
        List of stop words to be removed from the sentence
    lema : bool
        If True, lematization is applied
    stem : bool
        If True, stemming is applied

    Returns
    -------
    processed_string_list : list[str]
        A list of transformed strings.
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
    Apply some text transformations to a sentence. Remove stopwords, punctuation, stemming, lematization.

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


def get_vecinity_matrix_by_sentence(
        documents_list_with_matrix_positions: list[dict], 
        reference_terms: list[str], 
        limit_distance: int, 
        include_reference_term: bool,
        format_adjacent_refterms: bool
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
    format_adjacent_refterms : bool
        If True, format terms only if they are adjacent
    
    Returns
    -------
    vecinity_matrix : list[dict]
        A list of dictionaries, each dictionary contains the terms in the vecinity of the reference term and their
        corresponding frequencies by distances for each sentence
    """
    for document in documents_list_with_matrix_positions:
        vecinity_list = []
        for term_positions_defaultdict in document['text']:
            document_term_vecinity_dict = get_document_term_vecinity_dict(term_positions_defaultdict, reference_terms, limit_distance, 
                                                                          document['weight'], include_reference_term, format_adjacent_refterms)
            vecinity_list.append(document_term_vecinity_dict)
        document['text'] = vecinity_list

    return documents_list_with_matrix_positions


def get_document_term_vecinity_dict(
        term_positions_defaultdict: defaultdict[str, list[int]], 
        reference_terms: list[str], 
        limit_distance: int,
        doc_weight: float = 1.0,
        include_reference_terms: bool = True,
        format_adjacent_refterms: bool = True
        ) -> dict[str, dict[str, list[float]]]:
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
    doc_weight : float
        Weight of the document
    include_reference_terms : bool
        If True, the reference term is included in the vecinity
    format_adjacent_refterms : bool
        If True, format terms only if they are adjacent
    
    Returns
    -------
    vecinity_dict : dict[str, dict[str, list[float]]]
        A dictionary with the terms in the vecinity of the reference term and their corresponding distances
    """

    vecinity_dict = {}  # Create the empty dictionary
    ref_term_positions_dict = get_ref_term_positions_dict(term_positions_defaultdict, reference_terms, format_adjacent_refterms)
    
    # Calculate all terms in term_positions_defaultdict that are at distance limit_distance (or closer) to the reference_terms
    # and return a list of these terms and their corresponding distances
    for term, term_positions in term_positions_defaultdict.items():
        if((term not in ref_term_positions_dict.keys()) or (include_reference_terms)): #Avoid comparing the ref term with itself (if bool false)
            # Calculate the distance between the reference term and the rest of terms
            first_one = True
            for ref_term, ref_positions in ref_term_positions_dict.items():
                if ref_term != term:
                    freq_neighborhood_positions = calculate_frequency_term_positions_distances(ref_positions, term_positions, doc_weight, limit_distance)

                    if (any(frq > 0 for frq in freq_neighborhood_positions)):
                        if (first_one):
                            vecinity_dict[term] = {}
                            first_one = False
                        vecinity_dict[term][ref_term] = freq_neighborhood_positions

    return vecinity_dict


def get_ref_term_positions_dict(
        term_positions_defaultdict: defaultdict[str, list[int]], 
        reference_terms: list[str],
        format_adjacent_refterms: bool = True
        ) -> dict[str, list[int]]:
    """
    Returns a dictionary to store the list of positions for each reference term, along with its splitted terms.
    This function takes a defaultdict containing term positions and a list of reference terms as input.
    It splits each reference term into individual words and retrieves the positions of each term from the defaultdict.
    If a reference term contains multiple words, the function retrieves positions for each individual word and combines them.
    The resulting dictionary stores each reference term along with its list of positions.

    Parameters
    ----------
    term_positions_defaultdict : defaultdict[str, list[int]]
        A defaultdict containing term positions.
    reference_terms : list[str]
        A list of reference terms.
    format_adjacent_refterms: bool
        If True, format terms only if they are adjacent

    Returns
    -------
    ref_term_positions_dict : dict[str, list[int]
        A dictionary with reference terms as keys and lists of positions as values.
    """
    ref_term_positions_dict = {}

    for ref_term in reference_terms:
        ref_term_words = ref_term.split(' ')
        # If the reference term is within the term position dictionary
        if ref_term_words[0] in term_positions_defaultdict.keys():
            # If the reference term contains more than one word
            if (len(ref_term_words) > 1):
                for splitted_ref_term in ref_term_words:   
                    # Get the term positions of each splitted reference term
                    ref_term_positions_dict[splitted_ref_term] = term_positions_defaultdict[splitted_ref_term]
                if format_adjacent_refterms:
                    ref_term_positions_dict.update(format_adjacent_ref_term_positions_dict(ref_term_positions_dict, ref_term_words))
            # If the reference term contains just one word
            else:
                # Get the term positions of the reference term
                ref_term_positions_dict[ref_term] = term_positions_defaultdict[ref_term]    
    
    return ref_term_positions_dict


def format_adjacent_ref_term_positions_dict(
        ref_term_positions_dict: dict[str, list[int]],
        reference_term_words: list[str]
        ) -> dict[str, list[int]]:
    """
    Format the reference term positions dictionary, so that it can be used in the vecinity matrix.
    This function takes a reference term positions dictionary and a list of reference term words as input.
    It creates a new dictionary containing only the keys present in the reference term words list,
    along with their corresponding lists of positions.

    Parameters
    ----------
    ref_term_positions_dict : dict[str, list[int]]
        A dictionary containing term positions.
    reference_term_words : list[str]
        A list of reference term words.

    Returns
    -------
    formatted_dict : dict[str, list[int]]
        A formatted dictionary with reference term words as keys and lists of positions as values.
    """
    ref_term_positions_dict_splitted = defaultdict(list)

    #Omit keys that are not in reference_term_words
    for key, value in ref_term_positions_dict.items():
        if key in reference_term_words:
            ref_term_positions_dict_splitted[key] = value
    
    new_dict = get_new_dict_formatted_adjacent_ref_term_positions(reference_term_words, ref_term_positions_dict_splitted)
    formatted_dict = get_new_dict_formatted_adjacent_ref_term_positions(reference_term_words, new_dict)
    
    return formatted_dict


def get_new_dict_formatted_adjacent_ref_term_positions(
        reference_terms: list[str],
        ref_term_positions_dict: dict[str, list[int]]
        ) -> dict[str, list[int]]:
    """
    Get a new formatted reference term positions dictionary.
    This function takes a list of reference terms and a dictionary containing term positions as input.
    It iterates over the reference terms and retrieves the positions of adjacent terms in the reference terms list.
    The resulting dictionary contains each reference term word as a key and its list of positions as the value.

    Parameters
    ----------
    reference_terms : list[str]
        A list of reference terms.
    ref_term_positions_dict : dict[str, list[int]]
        A dictionary containing term positions.

    Returns
    -------
    new_formatted_dict : dict[str, list[int]]
        A new formatted dictionary with reference term words as keys and lists of positions as values.
    """
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
        doc_weight : float, 
        limit_distance: int = 99
        ) -> list[float]:
    """
    Compare the positions vectors of two terms, and return the frequency quantity list 
    per distance between query terms and vecinity terms

    Parameters
    ----------
    term1_positions : list[int]
        List of positions of the first term
    term2_positions : list[int]
        List of positions of the second term
    doc_weight : float
        Weight of the document
    limit_distance : float
        Maximal distance of terms

    Returns
    -------
    frequencies_per_distance : list[float]
        List of frequencies per distance between query terms and vecinity terms
    """
    frequencies_per_distance = [0] * limit_distance

    for term1_pos in term1_positions:
        for term2_pos in term2_positions:
            absolute_distance = abs(term1_pos-term2_pos)
            if (absolute_distance <= limit_distance):
                frequencies_per_distance[absolute_distance-1] += 1
    
    frequencies_per_distance = [i * doc_weight for i in frequencies_per_distance]
    return frequencies_per_distance


def get_unique_vecinity_matrix_by_document(
        vecinity_matrix: list[dict],
        limit_distance: int
        ) -> list[dict]:
    """
    Calculates a vecinity dictionary by document, merging their vecinities by sentence.

    Parameters
    ----------
    vecinity_matrix: list[dict]
        List of dictionaries, each dictionary contains the terms in the vecinity of the
        reference term and their corresponding frequency by distances
    limit_distance: int
        Maximal distance of terms

    Returns
    -------
    unique_vecinity_list : list[dict]
        A list of dictionaries with all its vecinities merged
    """
    for document in vecinity_matrix:
        # reduce() applies a function of two arguments cumulatively to the items of a sequence or iterable, from left to right, so 
        # as to reduce the iterable to a single value. For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5)
        unique_vecinity_document = reduce((lambda dict1, dict2: unite_dictionaries(dict1, dict2, limit_distance)), document['text'])
        document['text'] = unique_vecinity_document

    return vecinity_matrix


def unite_dictionaries(
        dict1: dict[str, dict[str, list[float]]], 
        dict2: dict[str, dict[str, list[float]]], 
        limit_distance: int = 99
        ) -> dict[str, dict[str, list[float]]]:
    """
    Unite two dictionaries into a new dictionary. 
    The merging process involves iterating through the keys of both dictionaries, summing the corresponding lists 
    for each subkey, and adding the keys that only exist in one dictionary. 
    The optional parameter limit_distance specifies the limit for the size of the lists to be united.
    
    Parameters
    ----------
    dict1 : dict[str, dict[str, list[float]]]
        The first dictionary to merge.
    dict2 : dict[str, dict[str, list[float]]]
        The second dictionary to merge.
    limit_distance : int
        Limit of distance for the merging process.

    Returns
    -------
    united_dict : dict[str, dict[str, list[float]]]
        A new dictionary with all its vecinities united.
    """
    united_dict = {}

    # Iterate over level 1 keys from the union between both dictionaries
    for key in dict1.keys() | dict2.keys():
        united_dict[key] = {}
        
        # Sum the corresponding lists, value by value, iterating in the union between the subkeys of the two dictionaries
        for subkey in dict1.get(key, {}).keys() | dict2.get(key, {}).keys():
            united_dict[key][subkey] = get_sum_of_freq_lists_by_distance(dict1, dict2, key, subkey, limit_distance)

    # Add level 1 keys that are not in both dictionaries
    for key in dict1.keys() - dict2.keys():
        united_dict[key] = dict1[key]

    for key in dict2.keys() - dict1.keys():
        united_dict[key] = dict2[key]

    return united_dict


def intersect_dictionaries(
        dict1: dict[str, dict[str, list[float]]], 
        dict2: dict[str, dict[str, list[float]]], 
        limit_distance: int = 99
        ) -> dict[str, dict[str, list[float]]]:
    """
    Intersect two dictionaries into a new dictionary. 
    The intersect process involves iterating through the keys that only exist both dictionaries, summing the 
    corresponding lists for each subkey.
    The optional parameter limit_distance specifies the limit for the size of the lists to be merged.
    
    Parameters
    ----------
    dict1 : dict[str, dict[str, list[float]]]
        The first dictionary to intersect.
    dict2 : dict[str, dict[str, list[float]]]
        The second dictionary to intersect.
    limit_distance : int
        Limit of distance for the intersecting process.

    Returns
    -------
    intersected_dict : dict[str, dict[str, list[float]]]
        A new dictionary with all its vecinities intersected.
    """
    intersected_dict = {}

    # Iterate over level 1 keys from the intersection between both dictionaries
    for key in dict1.keys() & dict2.keys():
        intersected_dict[key] = {}
        
        # Sum the corresponding lists, value by value, iterating in the union between the subkeys of the two dictionaries
        for subkey in dict1.get(key, {}).keys() | dict2.get(key, {}).keys():
            intersected_dict[key][subkey] = get_sum_of_freq_lists_by_distance(dict1, dict2, key, subkey, limit_distance)

    return intersected_dict


def get_sum_of_freq_lists_by_distance(
        dict1: dict[str, dict[str, list[float]]], 
        dict2: dict[str, dict[str, list[float]]], 
        key: str, 
        subkey: str, 
        limit_distance: int
        ) -> list[float]:
        list_of_freq_by_distance_dict1 = dict1.get(key, {}).get(subkey, [0] * limit_distance)
        list_of_freq_by_distance_dict2 = dict2.get(key, {}).get(subkey, [0] * limit_distance)
        sum_of_freq_lists_by_distance = [x + y for x, y in zip(list_of_freq_by_distance_dict1, list_of_freq_by_distance_dict2)]
        return sum_of_freq_lists_by_distance


def get_most_frequency_distance_list_by_refterm(
        unique_vecinity_list_by_doc: list[dict],
        reference_term: str,
        reference_terms: list[str],
        nr_of_graph_terms: int = 10
        ) -> list[dict[str, dict]]:
    """
    This method calculates the most frequency distance list based on the input parameters.

    Parameters
    ----------
    unique_vecinity_list_by_doc : list[dict]
        A list of dictionaries representing unique vicinity lists for each document. Each dictionary should have
        a key "text", containing a nested dictionary of terms and their frequencies within a document.
    reference_term : str
        The reference term used to calculate distances.
    reference_terms : list[str]
        List of reference terms from the query
    nr_of_graph_terms : int
        Configured number of terms in the graph
    
    Returns
    -------
    most_freq_distance_list : list[dict[str, dict]]
        A list of dictionaries containing the most frequency distance for each term.
        Each dictionary has keys corresponding to terms, and values are dictionaries with keys 'frequency' and 'distance',
        representing the frequency and mean distance of the term respectively.
    """
    first_sorted_terms_freq_list = get_first_sorted_terms_frequency_list(unique_vecinity_list_by_doc, nr_of_graph_terms, reference_term, reference_terms)
    most_freq_distance_list = []

    # Iterate over each term frequency dictionary
    for idx_freq_list, first_sorted_terms_freq_dict in enumerate(first_sorted_terms_freq_list):
        most_freq_distance_dict = {}

        # Iterate over each term in the frequency dictionary
        for fs_term in first_sorted_terms_freq_dict.keys():
            # Retrieve the list of frequencies by distance for the term
            list_of_freq_by_distance = unique_vecinity_list_by_doc[idx_freq_list].get("text").get(fs_term).get(reference_term)
            distance_calculation_list = []
            # Construct a list of distances multiplied by its frequencies
            for idx, freq_mult_by_weight in enumerate(list_of_freq_by_distance):
                weight_by_doc_idx = unique_vecinity_list_by_doc[idx_freq_list].get("weight")
                frequency = round(freq_mult_by_weight / weight_by_doc_idx)
                distance_calculation_list.extend([idx+1] * frequency)
                
             # Calculate the mean distance for the term
            most_freq_distance_dict[fs_term] = {'ponderation': first_sorted_terms_freq_dict.get(fs_term), 'distance': np.mean(distance_calculation_list)}
        
        # Append the dictionary for the current document to the result list
        most_freq_distance_list.append(most_freq_distance_dict)

    return most_freq_distance_list


def get_first_sorted_terms_frequency_list(
        unique_vecinity_list_by_doc: list[dict],
        nr_of_graph_terms: int,
        reference_term: str,
        reference_terms: list[str]
        ) -> list[dict[str, float]]:
    """
    Sort term frequency for each dictionary from a list of terms frequency in descending 
    order, limited by the configured number of terms in the graph

    Parameters
    ----------
    unique_vecinity_list_by_doc : list[dict]
        A list of dictionaries representing unique vicinity lists for each document. Each dictionary should
        have a key "text", containing a nested dictionary of terms and their frequencies within a document.
    nr_of_graph_terms : int
        Configured number of terms in the graph
    reference_term : str
        The reference term used to calculate distances.
    reference_terms : list[str]
        List of reference terms from the query

    Returns
    -------
    first_sorted_terms_freq_list : list[dict[str, float]]
        A list of first dictionaries sorted by their frequency per document. Each dictionary has keys 
        corresponding to terms, and values represent the frequency of the term within the document.

    """
    terms_freq_list = get_terms_frequency_list(unique_vecinity_list_by_doc, reference_term, reference_terms)
    first_sorted_terms_freq_list = [sort_dictionary_and_limit_by_number_of_graph_terms(dictionary, nr_of_graph_terms) 
                                    for dictionary in terms_freq_list]
    return first_sorted_terms_freq_list


def get_terms_frequency_list(
        unique_vecinity_list_by_doc: list[dict],
        reference_term: str,
        reference_terms: list[str]
        ) -> list[dict[str, float]]:
    """
    This method calculates the frequency of terms by a specified reference term in each dictionary from a 
    list of unique vicinity lists.

    Parameters
    ----------
    unique_vecinity_list_by_doc : list[dict]
        A list of dictionaries representing unique vicinity lists for each document. Each dictionary should
        have a key "text", containing a nested dictionary of terms and their frequencies within a document.
    reference_term : str
        The reference term used to calculate distances.
    reference_terms : list[str]
        List of reference terms from the query

    Returns
    -------
    terms_freq_list : list[dict[str, float]]
        A list of dictionaries containing the frequency of terms for each document. Each dictionary has keys 
        corresponding to terms, and values represent the frequency of the term within the document.

    """
    terms_freq_list = []

    # Iterate over each document's vicinity list
    for document in unique_vecinity_list_by_doc:
        terms_freq_dict = {}

        # Iterate over each term and its frequencies in the document
        for neighbor_term, freq_by_query_term in document["text"].items():
            # Split the reference terms into a list of words  (e.g. ["iot", "internet thing"] -> ["iot", "internet", "thing"])
            ref_terms_splitted = [word for string in reference_terms for word in string.split(" ")]  
            if (reference_term in freq_by_query_term) and (neighbor_term not in ref_terms_splitted):
                sum_freq_in_ref_term = sum(freq_by_query_term[reference_term])
                if sum_freq_in_ref_term > 0:
                    terms_freq_dict[neighbor_term] = sum_freq_in_ref_term
                
        # Append the term frequency dictionary for the current document to the result list        
        terms_freq_list.append(terms_freq_dict)
        
    return terms_freq_list


def sort_dictionary_and_limit_by_number_of_graph_terms(
        dictionary: dict[str, float],
        nr_of_graph_terms: int
        ) -> dict[str, float]:
    """Function to sort the dictionary by values in descending order, limited by the 
    configured number of terms in the graph"""
    first_sorted_terms_freq_dict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)[:nr_of_graph_terms]}
    return first_sorted_terms_freq_dict


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
        p_with = str(dic['ponderation'])
        g.node("'" +str(counter)+"'", keyword, fixedsize='true', width=node_size, penwidth=p_with, color=node_color)
        g.edge('0', "'" +str(counter)+"'", label=str(dic['distance']), len=str(dic['distance']))
        
    return g


def get_unique_vecinity_dict(
        unique_vecinity_dict: list[dict], 
        limit_distance: int
        ) -> dict[str, dict[str, list[float]]]:
    """
    Calculates a general vecinity dictionary, merging their vecinities by document.

    Parameters
    ----------
    unique_vecinity_dict: list[dict]
        List of dictionaries, each dictionary contains a vecinity by document
    limit_distance: int
        Maximal distance of terms

    Returns
    -------
    unique_vecinity_dict : dict[str, dict[str, list[float]]]
        A dictionary with all its vecinities merged
    """
    term_dict_list = []
    for dct in unique_vecinity_dict:
        term_dict_list.append(dct["text"])
    # reduce() applies a function of two arguments cumulatively to the items of a sequence or iterable, from left to right, so 
    # as to reduce the iterable to a single value. For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5)
    unique_vecinity_dict = reduce((lambda x, y: unite_dictionaries(x, y, limit_distance)), term_dict_list)
    return unique_vecinity_dict


def get_unique_graph_dictionary(graph_dictionaries_array):
    """Calculates a vecinity dictionary from an array of graph dictionaries, merging their vecinities."""
    product = reduce((lambda x, y: merge_graph_dictionaries(x,y)), graph_dictionaries_array)
    return product


def merge_graph_dictionaries(g1, g2):
    """Merge two "graph" dictionaries."""
    for i in g1.keys():
        if (i in g2.keys()):
            g2[i]['ponderation'] = g2[i]['ponderation'] + g1[i]['ponderation']
            g2[i]['distance'] = g2[i]['distance'] + g1[i]['distance']
        else:
            g2[i] = {'ponderation': g1[i]['ponderation'], 'distance': g1[i]['distance']} 
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


def getSummarizedGraph(self, summarize='median', normalize_frequency_range=False, normalize_distance_range=False):
        # Calculate a term distance metric for all nodes
        if (summarize =='median'):
            summarized_graph = {k: {'ponderation':self.nodes[k]['ponderation'], 
                                    'distance':np.median(self.nodes[k]['distances'])} for k in self.nodes.keys()}
        elif (summarize == 'mean'):
            summarized_graph = {k: {'ponderation':self.nodes[k]['ponderation'], 
                                    'distance':np.mean(self.nodes[k]['distances'])} for k in self.nodes.keys()}
        
        if((normalize_frequency_range != True) | (normalize_distance_range != False)):
            summarized_graph_distance = {k: v['distance'] for k, v in summarized_graph.items()}
            summarized_graph_frequency = {k: v['ponderation'] for k, v in summarized_graph.items()}
            
            # normalize distance
            if(normalize_distance_range != False):
                a1 = summarized_graph_distance[max(summarized_graph_distance, key=summarized_graph_distance.get)]
                c1 = summarized_graph_distance[min(summarized_graph_distance, key=summarized_graph_distance.get)]
                b1 = normalize_distance_range[1]
                d1 = normalize_distance_range[0]
                if((a1 - c1)>0):
                    m1 = (b1 - d1) / (a1 - c1)
                else:
                    m1 = (b1 - d1) # term frequency dictionary have only sigle words (frequency=1)
                summarized_graph_distance.update((k, (m1*(summarized_graph_distance[k]-c1)+d1)) for k in summarized_graph_distance.keys())
            
            # normalize frequency
            if(normalize_frequency_range != False):
                a2 = summarized_graph_frequency[max(summarized_graph_frequency, key=summarized_graph_frequency.get)]
                c2 = summarized_graph_frequency[min(summarized_graph_frequency, key=summarized_graph_frequency.get)]
                b2 = normalize_frequency_range[1]
                d2 = normalize_frequency_range[0]
                if((a2 - c2)>0):
                    m2 = (b2 - d2) / (a2 - c2)
                else:
                    m2 = (b2 - d2) # term frequency dictionary have only sigle words (frequency=1)
                summarized_graph_frequency.update((k, (m2*(summarized_graph_frequency[k]-c2)+d2)) for k in summarized_graph_frequency.keys())

            # Update the graph with normalized values
            summarized_graph = dict(map(self.__integrate, summarized_graph_frequency.items(), summarized_graph_distance.items()))
        
        return summarized_graph


# Calculate the similarity with otherGraph
# To calculate similarity we propose to compare the graph using a multidimensional
# vector space, where the each term properties define a dimension of the space.
def getSimilarity(self, otherGraph):
    # initialize the vectors
    u = [] 
    v = []
    # Calculate the base vector with terms of both vector (union)
    vectorBase = self.__getVectorBase(otherGraph)
    
    # Get the graphs versions with summarized distances
    self_sum = self.getSummarizedGraph()
    otherGraph_sum = otherGraph.getSummarizedGraph()
    
    # Calculate the vectors u and v in the multidimensional space
    # u: corresponds to self graph
    # v: correspnds to the otherGraph
    for term in vectorBase: # Generate de vector space for both attributes (ponderation and distance)
        if (term in self_sum):
            u.append(self_sum[term]['ponderation'])
            u.append(self_sum[term]['distance'])
        else:
            u.append(0) # ponderation value equal to cero
            u.append(0) # distance value equal to cero
        if (term in otherGraph_sum):
            v.append(otherGraph_sum[term]['ponderation'])
            v.append(otherGraph_sum[term]['distance'])
        else:
            v.append(0) # ponderation value equal to cero
            v.append(0) # distance value equal to cero

    # Calculate the cosine of the angle between the vectors
    cosine_of_angle = dot(u,v)/norm(u)/norm(v)
    
    return cosine_of_angle
