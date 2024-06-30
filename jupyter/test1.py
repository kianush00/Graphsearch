import unittest
import srex_new_classes as srex
import numpy as np
from collections import defaultdict
import json
import os

class TestSREX(unittest.TestCase):
    
    def test_union_between_graphs(self):
        # Initialize vicinity graphs
        g1 = srex.VicinityGraph(subquery="")
        g2 = srex.VicinityGraph(subquery="")
        r = srex.VicinityGraph(subquery="")

        # Build graph 1
        g1_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([3,4]))
        g1_node2 = srex.VicinityNode(term='y', ponderation=3.0, distance=np.mean([1,3,5]))
        g1.add_node(g1_node1)
        g1.add_node(g1_node2)

        # Build graph 2
        g2_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([1,2]))
        g2_node2 = srex.VicinityNode(term='y', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2_node3 = srex.VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2.add_node(g2_node1)
        g2.add_node(g2_node2)
        g2.add_node(g2_node3)

        # Build graph result to be evaluated
        r_node1 = srex.VicinityNode(term='x', ponderation=4.0, distance=np.mean([1, 2, 3, 4]))
        r_node2 = srex.VicinityNode(term='y', ponderation=7.0, distance=np.mean([11, 23, 52, 3, 1, 3, 5]))
        r_node3 = srex.VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        r.add_node(r_node1)
        r.add_node(r_node2)
        r.add_node(r_node3)

        # Define result as union between graph 1 and graph 2
        result = g1.get_union_to_graph(g2).get_graph_as_dict()
        expected_result = r.get_graph_as_dict()

        self.assertDictEqual(result, expected_result)
    

    def test_intersection_between_graphs(self):
        # Initialize vicinity graphs
        g1 = srex.VicinityGraph(subquery="")
        g2 = srex.VicinityGraph(subquery="")
        r = srex.VicinityGraph(subquery="")

        # Build graph 1
        g1_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([3,4]))
        g1_node2 = srex.VicinityNode(term='y', ponderation=3.0, distance=np.mean([1,3,5]))
        g1.add_node(g1_node1)
        g1.add_node(g1_node2)

        # Build graph 2
        g2_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([1,2]))
        g2_node2 = srex.VicinityNode(term='y', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2_node3 = srex.VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2.add_node(g2_node1)
        g2.add_node(g2_node2)
        g2.add_node(g2_node3)

        # Build graph result to be evaluated
        r_node1 = srex.VicinityNode(term='x', ponderation=4.0, distance=np.mean([1, 2, 3, 4]))
        r_node2 = srex.VicinityNode(term='y', ponderation=7.0, distance=np.mean([11, 23, 52, 3, 1, 3, 5]))
        r.add_node(r_node1)
        r.add_node(r_node2)

        # Define result as intersection between graph 1 and graph 2
        result = g1.get_intersection_to_graph(g2).get_graph_as_dict()
        expected_result = r.get_graph_as_dict()

        # Assert the result matches the expected output
        self.assertDictEqual(result, expected_result)
        

    def test_calculate_ponderation_of_distances_between_term_positions(self):
        # Create a Sentence object with an empty raw_text and a dummy query
        s = srex.Sentence(raw_text="", query=srex.BinaryExpressionTree(raw_query="test"))
        
        # Define term positions
        tp1 = [0, 2, 4, 6]
        tp2 = [1, 3, 5, 7]
        
        # Define limit distance
        limit_distance = 7
        
        # Calculate ponderation of distances between term positions
        result = s.calculate_ponderation_of_distances_between_term_positions(tp1, tp2, limit_distance)
        
        # Define expected result
        expected_result = [7.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0]
        
        # Assert the result matches the expected output
        self.assertCountEqual(result, expected_result)
    

    def test_remove_special_characters(self):
        # Load test text data
        test_data = self.__get_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_remove_special_characters')
        
        # Tokenize the text
        tokens = srex.nltk.word_tokenize(text.lower())
        
        # Remove special characters
        result = srex.TextUtils.remove_special_characters(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_remove_stopwords(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test text data
        test_data = self.__get_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_remove_stopwords')
        
        # Tokenize the text
        tokens = srex.nltk.word_tokenize(text.lower())
        
        # Remove stopwords
        result = srex.TextUtils.remove_stopwords(tokens, stop_words)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)

    
    def test_do_lemmatization(self):
        # Load test text data
        test_data = self.__get_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_do_lemmatization')
        
        # Tokenize the text
        tokens = srex.nltk.word_tokenize(text.lower())
        
        # Perform lemmatization
        result = srex.TextUtils.do_lemmatization(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_do_stemming(self):
        # Load test data and expected result from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_do_stemming')
        
        # Tokenize the text
        tokens = srex.nltk.word_tokenize(text.lower())
        
        # Perform stemming
        result = srex.TextUtils.do_stemming(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_get_transformed_text_with_lema(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data and expected result from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_transformed_text_with_lema')
        
        # Set lema and stem flags
        lema = True
        stem = False
        
        # Get transformed text
        result = srex.TextUtils.get_transformed_text(text, stop_words, lema, stem)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_get_transformed_text_if_it_has_underscores(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Initialize text variables
        text_with_underscores = "internet_of_things"
        expected_result = "internet_thing"
        
        # Set lema and stem flags
        lema = True
        stem = False
        
        # Get transformed text
        result = srex.TextUtils.get_transformed_text_if_it_has_underscores(text_with_underscores, stop_words, lema, stem)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
        

    def test_term_positions_dict(self):
        # Load test data from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text6')
        dict_result = test_data.get('text6_term_positions_dict')

        # Initialize Ranking object
        ranking = srex.Ranking(query_text="test")
        ranking.calculate_article_dictionaries_list([{'title': 'test'}])

        # Get term positions dictionary from ranking object
        result = ranking.get_documents()[0].get_sentences()[0].get_term_positions_dict(text)
        
        # Create expected result as defaultdict
        expected_result = defaultdict(list, dict_result)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_query_term_positions_dict(self):
        # Load term positions dictionaries from JSON file
        test_data = self.__get_test_data()
        term_positions_dict = test_data.get('text6_term_positions_dict')
        expected_result = test_data.get('text6_query_term_positions_dict')

        # Initialize Ranking object
        ranking = srex.Ranking(query_text="test")
        ranking.calculate_article_dictionaries_list([{'title': 'test'}])

        # Initialize query terms
        query_terms = ["driven", "adopt", "store"]

        # Get query term positions dictionary from ranking object
        result = ranking.get_documents()[0].get_sentences()[0].get_query_term_positions_dict(term_positions_dict, query_terms)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_calculate_vicinity_matrix_with_include_query_terms(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text6')
        expected_result = test_data.get('text6_calculate_vicinity_matrix_with_include_query_terms')
        article_dict = {'abstract': text}

        # Initialize Ranking object with include_query_terms as true
        query = 'driven OR adopt AND store'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()

        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms)
        
        #Calculate the vicinity matrix for the sentence
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.calculate_vicinity_matrix()
        result = sentence.get_vicinity_matrix()

        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_calculate_vicinity_matrix_without_include_query_terms(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text6')
        expected_result = test_data.get('text6_calculate_vicinity_matrix_without_include_query_terms')
        article_dict = {'abstract': text}

        # Initialize Ranking object with include_query_terms as false
        query = 'driven OR adopt AND store'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()
        include_query_terms = False

        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms)
        
        #Calculate the vicinity matrix for the sentence
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.calculate_vicinity_matrix()
        result = sentence.get_vicinity_matrix()

        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_get_terms_ponderation_dict(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_terms_ponderation_dict1')
        expected_result2 = test_data.get('text6_terms_ponderation_dict2')
        expected_result3 = test_data.get('text6_terms_ponderation_dict3')
        article_dict = {'abstract': text}

        # Initialize Ranking object
        query = 'driven OR adopt AND store'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()

        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        #Calculate the terms ponderation dict for the sentence, by each query term
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        query_terms = sentence.get_query_tree().get_query_terms_str_list_with_underscores()

        result1 = sentence.get_terms_ponderation_dict(query_terms[0])
        result2 = sentence.get_terms_ponderation_dict(query_terms[1])
        result3 = sentence.get_terms_ponderation_dict(query_terms[2])

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    

    def test_generate_nodes_in_all_leaf_graphs_summarize_mean(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_leaf_graph_mean_1')
        expected_result2 = test_data.get('text6_leaf_graph_mean_2')
        expected_result3 = test_data.get('text6_leaf_graph_mean_3')
        article_dict = {'abstract': text}

        # Initialize Ranking object with summarize as 'mean'
        query = 'driven OR adopt AND store'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()

        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        # Generate nodes in all graphs in leaf nodes of the query expression tree
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.generate_nodes_in_all_leaf_graphs()

        # Get vicinity graphs from the sentence, as dicts
        query_terms = sentence.get_query_tree().get_query_terms_str_list_with_underscores()
        result1 = sentence.get_graph_by_subquery(query_terms[0]).get_graph_as_dict()
        result2 = sentence.get_graph_by_subquery(query_terms[1]).get_graph_as_dict()
        result3 = sentence.get_graph_by_subquery(query_terms[2]).get_graph_as_dict()

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    

    def test_generate_nodes_in_all_leaf_graphs_summarize_median(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_leaf_graph_median_1')
        expected_result2 = test_data.get('text6_leaf_graph_median_2')
        expected_result3 = test_data.get('text6_leaf_graph_median_3')
        article_dict = {'abstract': text}

        # Initialize Ranking object with summarize as 'median'
        query = 'driven OR adopt AND store'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()
        summarize = 'median'

        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        # Generate nodes in all graphs in leaf nodes of the query expression tree
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.generate_nodes_in_all_leaf_graphs()

        # Get vicinity graphs from the sentence, as dicts
        query_terms = sentence.get_query_tree().get_query_terms_str_list_with_underscores()
        result1 = sentence.get_graph_by_subquery(query_terms[0]).get_graph_as_dict()
        result2 = sentence.get_graph_by_subquery(query_terms[1]).get_graph_as_dict()
        result3 = sentence.get_graph_by_subquery(query_terms[2]).get_graph_as_dict()

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    

    def test_generate_nodes_in_sentence_graphs(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_subgraph1')
        expected_result2 = test_data.get('text6_subgraph2')
        article_dict = {'abstract': text}

        # Initialize Ranking object with summarize as 'mean'
        query = 'driven OR adopt AND store'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()

        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        # Generate nodes in all graphs of the query expression tree
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.generate_nodes_in_all_leaf_graphs()
        sentence.get_query_tree().operate_graphs_from_leaves()

        # Get vicinity graphs from the subqueries, as dicts
        subquery_list = [
            'adopt AND store', 
            'driven OR (adopt AND store)'
        ]
        result1 = sentence.get_graph_by_subquery(subquery_list[0]).get_graph_as_dict()
        result2 = sentence.get_graph_by_subquery(subquery_list[1]).get_graph_as_dict()

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
    

    def test_get_union_to_tree(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text6 = test_data.get('text6')
        text2 = test_data.get('text2')
        expected_result1 = test_data.get('text6_text2_united_graph_1')
        expected_result2 = test_data.get('text6_text2_united_graph_2')
        expected_result3 = test_data.get('text6_text2_united_graph_3')
        article_dict = {'title': text6, 'abstract': text2}

        # Initialize Ranking object with summarize as 'mean'
        query = 'driven OR adopt AND store'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()

        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        # Generate nodes in all graphs of the query expression tree, in the sentences
        document = ranking.get_document_by_ranking_position(1)
        for sentence in document.get_sentences():
            sentence.generate_nodes_in_sentence_graphs()

        # Get union between two trees
        query_trees_list = document.get_list_of_query_trees_from_sentences()
        united_tree = query_trees_list[0].get_union_to_tree(query_trees_list[1])

        # Get united vicinity graphs from the subqueries, as dicts
        subquery_list = [
            'driven',
            'adopt AND store', 
            'driven OR (adopt AND store)'
        ]
        result1 = united_tree.get_graph_by_subquery(subquery_list[0]).get_graph_as_dict()
        result2 = united_tree.get_graph_by_subquery(subquery_list[1]).get_graph_as_dict()
        result3 = united_tree.get_graph_by_subquery(subquery_list[2]).get_graph_as_dict()

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    

    def test_generate_all_graphs(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text1 = test_data.get('text1')
        text2 = test_data.get('text2')
        text3 = test_data.get('text3')
        text4 = test_data.get('text4')
        expected_result = test_data.get('text_1_2_3_4_united_root_graph')

        list_of_articles_dicts = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        # Initialize Ranking object and generate all graphs
        query = 'network'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()
        
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, 
                               lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list(list_of_articles_dicts)
        ranking.generate_all_graphs(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)

        # Get the ranking root graph
        result = ranking.get_graph().get_graph_as_dict()

        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_cosine_similarity(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text1 = test_data.get('text1')
        text2 = test_data.get('text2')
        text3 = test_data.get('text3')
        text4 = test_data.get('text4')

        list_of_articles_dicts = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        # Initialize Ranking object
        query = 'network'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()
        
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, 
                               lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list(list_of_articles_dicts)
        ranking.generate_all_graphs(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)

        # Get cosine similarities between vicinity graphs from the documents
        graph1 = ranking.get_document_by_ranking_position(1).get_graph()
        graph2 = ranking.get_document_by_ranking_position(2).get_graph()
        graph3 = ranking.get_document_by_ranking_position(3).get_graph()
        graph4 = ranking.get_document_by_ranking_position(4).get_graph()

        result1 = graph1.get_cosine_similarity(graph1)
        result2 = graph1.get_cosine_similarity(graph2)
        result3 = graph1.get_cosine_similarity(graph3)
        result4 = graph1.get_cosine_similarity(graph4)

        # Create expected results
        expected_result1 = 1.0
        expected_result2 = 0.9671707945560485
        expected_result3 = 0.6557450996618941
        expected_result4 = 0.5949574537482146

        # Assert the result matches the expected output
        self.assertAlmostEqual(result1, expected_result1, delta=1e-14)
        self.assertAlmostEqual(result2, expected_result2, delta=1e-14)
        self.assertAlmostEqual(result3, expected_result3, delta=1e-14)
        self.assertAlmostEqual(result4, expected_result4, delta=1e-14)
    

    def test_get_terms_from_nodes(self):
        # Load stopwords from JSON file
        stop_words = self.__get_loaded_stopwords()

        # Load test data from JSON file
        test_data = self.__get_test_data()
        text1 = test_data.get('text1')
        text2 = test_data.get('text2')
        text3 = test_data.get('text3')
        text4 = test_data.get('text4')
        expected_result1 = test_data.get('text1_terms_from_nodes')
        expected_result2 = test_data.get('text2_terms_from_nodes')
        expected_result3 = test_data.get('text3_terms_from_nodes')
        expected_result4 = test_data.get('text4_terms_from_nodes')

        list_of_articles_dicts = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        # Initialize Ranking object and initialize all graphs
        query = 'network'
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_default_ranking_parameters()
        
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, 
                               lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list(list_of_articles_dicts)
        ranking.generate_all_graphs(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)

        # Get terms from nodes of the vicinity graphs associated with the documents
        result1 = ranking.get_document_by_ranking_position(1).get_graph().get_terms_from_nodes()
        result2 = ranking.get_document_by_ranking_position(2).get_graph().get_terms_from_nodes()
        result3 = ranking.get_document_by_ranking_position(3).get_graph().get_terms_from_nodes()
        result4 = ranking.get_document_by_ranking_position(4).get_graph().get_terms_from_nodes()

        # Assert the result matches the expected output
        self.assertCountEqual(result1, expected_result1)
        self.assertCountEqual(result2, expected_result2)
        self.assertCountEqual(result3, expected_result3)
        self.assertCountEqual(result4, expected_result4)
    

    def __get_default_ranking_parameters(self):
        ranking_weight_type = 'linear'  # it can be: 'none', 'linear' or 'inverse'
        lema = True
        stem = False
        summarize = 'mean'  # it can be: 'mean' or 'median'
        limit_distance = 4
        include_query_terms = True
        return ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms
    

    def __get_loaded_stopwords(self):
        stopwords_data_path = 'jupyter/json_data/stopwords_data.json'

        # Validate if the path exists
        if not os.path.exists(stopwords_data_path):
            raise FileNotFoundError(f"File '{stopwords_data_path}' does not exist.")

        with open(stopwords_data_path) as f:
            stopwords_data = json.load(f)
        stop_words = stopwords_data.get('words')
        return stop_words
    

    def __get_test_data(self):
        test_data_path = 'jupyter/json_data/test_data.json'

        # Validate if the path exists
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"File '{test_data_path}' does not exist.")

        with open(test_data_path) as f:
            test_data = json.load(f)
        return test_data


if __name__ == '__main__':
    unittest.main()