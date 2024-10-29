import unittest
from models.srex.binary_expression_tree import BinaryExpressionTree
from models.srex.ranking import Ranking
from models.srex.ranking import Sentence
from models.srex.vicinity_graph import VicinityGraph
from models.srex.vicinity_graph import VicinityNode
from utils.text_utils import TextUtils
from utils.vector_utils import VectorUtils
from utils.data_utils import DataUtils
import json

import nltk
import numpy as np
from collections import defaultdict


class TestSREX(unittest.TestCase):

    def test_binary_expression_tree_query(self):
        # Initialize binary expression tree query
        query = '"internet of things" OR iot NOT graph'
        b_expr_tree = BinaryExpressionTree(query)

        result = str(b_expr_tree)
        expected_result = 'internet_of_things OR iot'

        # Assert the query string matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_binary_expression_tree_query_text_transformations_01(self):
        # Initialize binary expression tree from query and do text transformations to the tree
        query = '"internet of things" OR iot NOT graph'
        b_expr_tree = self.__initialize_binary_expression_tree_with_text_transformations(query)

        result = str(b_expr_tree)
        expected_result = 'internet_thing OR iot'

        # Assert the query string matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_binary_expression_tree_query_text_transformations_02(self):
        # Initialize binary expression tree from query and do text transformations to the tree
        query = '("Document Title":internet of things   OR   ("Document Title":iot  AND "Document Title":device  )  )  AND ("Abstract":security NOT  "Abstract":visual OR "Document Title":network)'
        b_expr_tree = self.__initialize_binary_expression_tree_with_text_transformations(query)

        result = str(b_expr_tree)
        expected_result = '(internet_thing OR (iot AND device)) AND (security OR network)'

        # Assert the query string matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_binary_expression_tree_query_text_transformations_03(self):
        # Initialize binary expression tree from query and do text transformations to the tree
        query = '("Document Title":internet of things OR "Document Title":iot) NOT ("Abstract":visual OR "Abstract":retrieval) AND ("All Metadata":security)'
        b_expr_tree = self.__initialize_binary_expression_tree_with_text_transformations(query)

        result = str(b_expr_tree)
        expected_result = '(internet_thing OR iot) AND security'
        
        # Assert the query string matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_binary_expression_tree_query_text_transformations_04(self):
        # Initialize binary expression tree from query and do text transformations to the tree
        query = query = '((( literature  OR document OR information OR data ) AND (retrieval OR retrieve)) OR (search AND engine)  )  AND  (query AND  ( expansion OR refinement OR reformulation))'
        b_expr_tree = self.__initialize_binary_expression_tree_with_text_transformations(query)

        result = str(b_expr_tree)
        expected_result = '(((((literature OR document) OR information) OR data) AND (retrieval OR retrieve)) OR (search AND engine)) AND (query AND ((expansion OR refinement) OR reformulation))'

        # Assert the query string matches the expected output
        self.assertEqual(result, expected_result)


    def test_get_graph_as_dict(self):
        # Initialize vicinity graph
        graph = self.__get_initialized_graph_config_01()

        # Get the graph as a dictionary
        result = graph.get_graph_as_dict()

        # Define expected result
        expected_result = {
            'x': {'ponderation': 2.0, 'distance': 1.5}, 
            'y': {'ponderation': 3.0, 'distance': 2.5}, 
            'z': {'ponderation': 4.0, 'distance': 2.1}
        }

        # Assert the dict match the expected output
        self.assertDictEqual(result, expected_result)

    
    def test_get_proximity_dict_with_normalized_distances(self):
        # Initialize vicinity graph
        graph = self.__get_initialized_graph_config_01()

        # Get the graph as a normalized dictionary
        result = graph.get_proximity_dict_with_normalized_distances()

        # Define expected result
        expected_result = {
            'x': {'criteria': 'proximity', 'ponderation': 0.5, 'distance': 0.5833333333333334}, 
            'y': {'criteria': 'proximity', 'ponderation': 0.75, 'distance': 0.75}, 
            'z': {'criteria': 'proximity', 'ponderation': 1.0, 'distance': 0.6833333333333333}
        }

        # Assert the dict match the expected output
        self.assertDictEqual(result, expected_result)
    

    def test_normalize_vector(self):
        # Initialize vector values and new range
        vector = [1.0, 2.5, 4.0, 1.75, 3.25]
        new_min = 0.5
        new_max = 1.0

        # Normalize vector and assert the results
        result = VectorUtils.normalize_vector(vector, new_min, new_max)
        expected_result = [0.5, 0.75, 1.0, 0.625, 0.875]
        
        self.assertListEqual(result, expected_result)
    

    def test_get_cosine_between_vectors(self):
        # Initialize vectors values
        vector1 = [1.2, 2.3, 3.5, 3.0]
        vector2 = [3.1, 2.4, 1.1, 1.8]

        # Get the euclidean distance between vectors and assert the results
        result = VectorUtils.get_cosine_between_vectors(vector1, vector2)
        expected_result = 3.2893768406797053
        
        self.assertAlmostEqual(result, expected_result, delta=1e-13)

    

    def test_union_between_graphs(self):
        # Initialize vicinity graphs
        g1, g2 = self.__initialize_graph_1_and_2()
        r = VicinityGraph(subquery="")

        # Build graph result to be evaluated
        r_node1 = VicinityNode(term='x', ponderation=4.0, distance=np.mean([1, 2, 3, 4]))
        r_node2 = VicinityNode(term='y', ponderation=7.0, distance=np.mean([11, 23, 52, 3, 1, 3, 5]))
        r_node3 = VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        r.add_node(r_node1)
        r.add_node(r_node2)
        r.add_node(r_node3)

        # Define result as union between graph 1 and graph 2
        result = g1.get_union_to_graph(g2).get_graph_as_dict()
        expected_result = r.get_graph_as_dict()

        # Assert the dict matches the expected output
        self.assertDictEqual(result, expected_result)
    

    def test_intersection_between_graphs(self):
        # Initialize vicinity graphs
        g1, g2 = self.__initialize_graph_1_and_2()
        r = VicinityGraph(subquery="")

        # Build graph result to be evaluated
        r_node1 = VicinityNode(term='x', ponderation=4.0, distance=np.mean([1, 2, 3, 4]))
        r_node2 = VicinityNode(term='y', ponderation=7.0, distance=np.mean([11, 23, 52, 3, 1, 3, 5]))
        r.add_node(r_node1)
        r.add_node(r_node2)

        # Define result as intersection between graph 1 and graph 2
        result = g1.get_intersection_to_graph(g2).get_graph_as_dict()
        expected_result = r.get_graph_as_dict()

        # Assert the dict matches the expected output
        self.assertDictEqual(result, expected_result)
        

    def test_calculate_distances_between_term_positions(self):
        # Define term positions
        tp1 = [0, 2, 4, 6]
        tp2 = [1, 3, 5, 7]
        
        # Define limit distance
        limit_distance = 7
        
        # Calculate distances between term positions
        result = VectorUtils.calculate_distances_between_term_positions(tp1, tp2, limit_distance)
        
        # Define expected result
        expected_result = [7, 0, 5, 0, 3, 0, 1]
        
        # Assert the result matches the expected output
        self.assertCountEqual(result, expected_result)
    

    def test_remove_special_characters(self):
        # Load test text data
        test_data = DataUtils.load_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_remove_special_characters')
        
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove special characters
        result = TextUtils.remove_special_characters(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_remove_stopwords(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test text data
        test_data = DataUtils.load_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_remove_stopwords')
        
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stopwords
        result = TextUtils.remove_stopwords(tokens, stop_words)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)

    
    def test_do_lemmatization(self):
        # Load test text data
        test_data = DataUtils.load_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_do_lemmatization')
        
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        
        # Perform lemmatization
        result = TextUtils.do_lemmatization(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_do_stemming(self):
        # Load test data and expected result from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_do_stemming')
        
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        
        # Perform stemming
        result = TextUtils.do_stemming(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_get_transformed_text_with_lema(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data and expected result from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text5')
        expected_result = test_data.get('text5_transformed_text_with_lema')
        
        # Set lema and stem flags
        lema = True
        stem = False
        
        # Get transformed text
        result = TextUtils.get_transformed_text(text, stop_words, lema, stem)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_get_transformed_text_if_it_has_underscores(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Initialize text variables
        text_with_underscores = "internet_of_things"
        expected_result = "internet_thing"
        
        # Set lema and stem flags
        lema = True
        stem = False
        
        # Get transformed text
        result = TextUtils.get_transformed_text_if_it_has_underscores(text_with_underscores, stop_words, lema, stem)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
        

    def test_term_positions_dict(self):
        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text6')
        dict_result = test_data.get('text6_term_positions_dict')

        # Initialize Ranking object
        ranking = Ranking(query_text="test")
        ranking.calculate_article_dictionaries_list([{'title': 'test'}])

        # Get term positions dictionary from ranking object
        result = ranking.get_documents()[0].get_sentences()[0].get_term_positions_dict(text)
        
        # Create expected result as defaultdict
        expected_result = defaultdict(list, dict_result)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_query_term_positions_dict(self):
        # Load term positions dictionaries from JSON file
        test_data = DataUtils.load_test_data()
        term_positions_dict = test_data.get('text6_term_positions_dict')
        expected_result = test_data.get('text6_query_term_positions_dict')

        # Initialize Ranking object
        ranking = Ranking(query_text="test")
        ranking.calculate_article_dictionaries_list([{'title': 'test'}])

        # Initialize query terms
        query_terms = ["driven", "adopt", "store"]

        # Get query term positions dictionary from ranking object
        result = ranking.get_documents()[0].get_sentences()[0].get_query_term_positions_dict(term_positions_dict, query_terms)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_calculate_vicinity_matrix_with_include_query_terms(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text6')
        expected_result = test_data.get('text6_calculate_vicinity_matrix_with_include_query_terms')
        article_dict = {'abstract': text}

        # Initialize Ranking object with include_query_terms as true
        query = 'driven OR adopt AND store'
        ranking = self.__get_initialized_ranking_initialized_graph_values_01(
            self.__get_ranking_parameters_default_config, query, [article_dict], stop_words)
        
        #Calculate the vicinity matrix for the sentence
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.calculate_vicinity_matrix()
        result = sentence.get_vicinity_matrix()

        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_calculate_vicinity_matrix_without_include_query_terms(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text6')
        expected_result = test_data.get('text6_calculate_vicinity_matrix_without_include_query_terms')
        article_dict = {'abstract': text}

        # Initialize Ranking object with include_query_terms as false
        query = 'driven OR adopt AND store'
        ranking = self.__get_initialized_ranking_initialized_graph_values_01(
            self.__get_ranking_parameters_config_02, query, [article_dict], stop_words)
        
        #Calculate the vicinity matrix for the sentence
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.calculate_vicinity_matrix()
        result = sentence.get_vicinity_matrix()

        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_get_terms_proximity_frequency_dict(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_terms_frequency_dict1')
        expected_result2 = test_data.get('text6_terms_frequency_dict2')
        expected_result3 = test_data.get('text6_terms_frequency_dict3')
        article_dict = {'abstract': text}

        # Initialize Ranking object
        query = 'driven OR adopt AND store'
        ranking = self.__get_initialized_ranking_initialized_graph_values_01(
            self.__get_ranking_parameters_default_config, query, [article_dict], stop_words)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        #Calculate the terms frequency dict for the sentence, by each query term
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        query_terms = sentence.get_query_tree().get_query_terms_str_list_with_underscores()

        result1 = sentence.get_terms_proximity_frequency_dict(query_terms[0])
        result2 = sentence.get_terms_proximity_frequency_dict(query_terms[1])
        result3 = sentence.get_terms_proximity_frequency_dict(query_terms[2])

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    

    def test_generate_nodes_in_all_leaf_graphs_summarize_mean(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_leaf_graph_mean_1')
        expected_result2 = test_data.get('text6_leaf_graph_mean_2')
        expected_result3 = test_data.get('text6_leaf_graph_mean_3')
        article_dict = {'abstract': text}

        # Initialize Ranking object with summarize as 'mean'
        query = 'driven OR adopt AND store'
        ranking = self.__get_initialized_ranking_initialized_graph_values_01(
            self.__get_ranking_parameters_default_config, query, [article_dict], stop_words)
        
        result1, result2, result3 = self.__get_results_from_generate_nodes_in_all_leaf_graphs(ranking)

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    

    def test_generate_nodes_in_all_leaf_graphs_summarize_median(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_leaf_graph_median_1')
        expected_result2 = test_data.get('text6_leaf_graph_median_2')
        expected_result3 = test_data.get('text6_leaf_graph_median_3')
        article_dict = {'abstract': text}

        # Initialize Ranking object with summarize as 'median'
        query = 'driven OR adopt AND store'
        ranking = self.__get_initialized_ranking_initialized_graph_values_01(
            self.__get_ranking_parameters_config_03, query, [article_dict], stop_words)
        
        result1, result2, result3 = self.__get_results_from_generate_nodes_in_all_leaf_graphs(ranking)

        # Assert the result matches the expected output
        self.assertEqual(result1, expected_result1)
        self.assertEqual(result2, expected_result2)
        self.assertEqual(result3, expected_result3)
    

    def test_generate_nodes_in_sentence_graphs(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text6')
        expected_result1 = test_data.get('text6_subgraph1')
        expected_result2 = test_data.get('text6_subgraph2')
        article_dict = {'abstract': text}

        # Initialize Ranking object with summarize as 'mean'
        query = 'driven OR adopt AND store'
        ranking = self.__get_initialized_ranking_initialized_graph_values_01(
            self.__get_ranking_parameters_default_config, query, [article_dict], stop_words)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        # Generate nodes in all graphs of the query expression tree
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.generate_nodes_in_all_leaf_graphs()
        sentence.get_query_tree().operate_non_leaf_graphs_from_leaves()

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
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text6 = test_data.get('text6')
        text2 = test_data.get('text2')
        expected_result1 = test_data.get('text6_text2_united_graph_1')
        expected_result2 = test_data.get('text6_text2_united_graph_2')
        expected_result3 = test_data.get('text6_text2_united_graph_3')
        article_dict = {'title': text6, 'abstract': text2}

        # Initialize Ranking object with summarize as 'mean'
        query = 'driven OR adopt AND store'
        ranking = self.__get_initialized_ranking_initialized_graph_values_01(
            self.__get_ranking_parameters_default_config, query, [article_dict], stop_words)
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        # Generate nodes in all graphs of the query expression tree, in the sentences
        document = ranking.get_document_by_ranking_position(1)
        for sentence in document.get_sentences():
            sentence.generate_nodes_in_tree_graphs()

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
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        expected_result = test_data.get('text_1_2_3_4_united_root_graph')
        
        # Initialize Ranking object and generate all graphs
        ranking = self.__get_initialized_ranking_config_01(test_data, stop_words)

        # Get the ranking root graph
        result = ranking.get_graph().get_graph_as_dict()

        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_euclidean_distance_exclude_ponderation(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        
        # Initialize Ranking object and generate all graphs
        ranking = self.__get_initialized_ranking_config_01(test_data, stop_words)

        # Get euclidean distances between vicinity graphs from the documents
        graph1 = ranking.get_document_by_ranking_position(1).get_graph()
        graph2 = ranking.get_document_by_ranking_position(2).get_graph()
        graph3 = ranking.get_document_by_ranking_position(3).get_graph()
        graph4 = ranking.get_document_by_ranking_position(4).get_graph()

        result1 = graph1.get_similarity_score_as_base_graph(graph1)
        result2 = graph1.get_similarity_score_as_base_graph(graph2)
        result3 = graph1.get_similarity_score_as_base_graph(graph3)
        result4 = graph1.get_similarity_score_as_base_graph(graph4)

        # Create expected results
        expected_result1 = 0.0
        expected_result2 = 0.2041241452319315
        expected_result3 = 0.30046260628866567
        expected_result4 = 0.24999999999999992

        # Assert the result matches the expected output
        self.assertAlmostEqual(result1, expected_result1, delta=1e-13)
        self.assertAlmostEqual(result2, expected_result2, delta=1e-13)
        self.assertAlmostEqual(result3, expected_result3, delta=1e-13)
        self.assertAlmostEqual(result4, expected_result4, delta=1e-13)
    

    def test_euclidean_distance_include_ponderation(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        
        # Initialize Ranking object and generate all graphs
        ranking = self.__get_initialized_ranking_config_01(test_data, stop_words)

        # Get euclidean distances between vicinity graphs from the documents
        graph1 = ranking.get_document_by_ranking_position(1).get_graph()
        graph2 = ranking.get_document_by_ranking_position(2).get_graph()
        graph3 = ranking.get_document_by_ranking_position(3).get_graph()
        graph4 = ranking.get_document_by_ranking_position(4).get_graph()

        result1 = graph1.get_similarity_score_as_base_graph(graph1, True)
        result2 = graph1.get_similarity_score_as_base_graph(graph2, True)
        result3 = graph1.get_similarity_score_as_base_graph(graph3, True)
        result4 = graph1.get_similarity_score_as_base_graph(graph4, True)

        # Create expected results
        expected_result1 = 0.0
        expected_result2 = 0.2041241452319315
        expected_result3 = 0.30046260628866567
        expected_result4 = 0.5590169943749473

        # Assert the result matches the expected output
        self.assertAlmostEqual(result1, expected_result1, delta=1e-13)
        self.assertAlmostEqual(result2, expected_result2, delta=1e-13)
        self.assertAlmostEqual(result3, expected_result3, delta=1e-13)
        self.assertAlmostEqual(result4, expected_result4, delta=1e-13)
    

    def test_get_terms_from_nodes(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text4')
        expected_result = test_data.get('text4_terms_from_nodes')
        
        # Initialize Ranking object and initialize vicinity graph
        query = 'network'
        ranking = self.__get_initialized_ranking_generated_all_graphs_01(
            self.__get_ranking_parameters_default_config, query, articles_dicts_list=[{'abstract': text}], stop_words=stop_words)

        # Get terms from nodes of the vicinity graphs associated with the document
        result = ranking.get_document_by_ranking_position(1).get_graph().get_terms_from_nodes()

        # Assert the result matches the expected output
        self.assertListEqual(result, expected_result)
    

    def test_get_viewable_graph_copy(self):
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Load test data from JSON file
        test_data = DataUtils.load_test_data()
        text = test_data.get('text4')
        expected_result = test_data.get('text4_terms_from_viewable_graph_copy')
        
        # Initialize Ranking object and initialize all graphs
        query = 'network OR hierarchical'
        ranking = self.__get_initialized_ranking_generated_all_graphs_01(
            self.__get_ranking_parameters_default_config, query, articles_dicts_list=[{'abstract': text}], stop_words=stop_words)

        # Get terms from nodes of the viewable copy from the vicinity graph
        graph = ranking.get_document_by_ranking_position(1).get_graph()
        viewable_graph_copy = graph.get_viewable_graph_copy()
        result = viewable_graph_copy.get_terms_from_nodes()

        # Assert the result matches the expected output
        self.assertListEqual(result, expected_result)


    

    ############## PRIVATE FUNCTIONS ##############

    def __get_initialized_ranking_config_01(self, test_data, stop_words: tuple[str] = []) -> Ranking:
        text1 = test_data.get('text1')
        text2 = test_data.get('text2')
        text3 = test_data.get('text3')
        text4 = test_data.get('text4')

        articles_dicts_list = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        # Initialize Ranking object and generate all graphs
        query = 'network'
        ranking = self.__get_initialized_ranking_generated_all_graphs_01(
            self.__get_ranking_parameters_default_config, query, articles_dicts_list, stop_words)

        return ranking
    

    def __get_initialized_ranking_generated_all_graphs_01(
            self, method_to_call, query: str, articles_dicts_list: list[dict[str, str]], stop_words: tuple[str] = []) -> Ranking:
        # Initialize Ranking object and generate all graphs
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = method_to_call()
        
        ranking = Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list(articles_dicts_list)
        ranking.generate_all_graphs(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)

        return ranking
    

    def __get_initialized_ranking_initialized_graph_values_01(
            self, method_to_call, query: str, articles_dicts_list: list[dict[str, str]], stop_words: tuple[str] = []) -> Ranking:
        # Initialize Ranking object
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = method_to_call()

        ranking = Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list(articles_dicts_list)
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms, summarize=summarize)

        return ranking
    

    def __get_initialized_graph_config_01(self) -> VicinityGraph:
        # Initialize vicinity graph
        graph = VicinityGraph(subquery="")

        # Build graph
        node1 = VicinityNode(term='x', ponderation=2.0, distance=np.mean([1.0, 2.0]))
        node2 = VicinityNode(term='y', ponderation=3.0, distance=np.mean([1.2, 2.3, 3.5, 3.0]))
        node3 = VicinityNode(term='z', ponderation=4.0, distance=np.mean([3.1, 2.4, 1.1, 1.8]))
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        return graph
    

    def __get_ranking_parameters_default_config(self):
        ranking_weight_type = 'linear'  # it can be: 'none', 'linear' or 'inverse'
        lema = True
        stem = False
        summarize = 'mean'  # it can be: 'mean' or 'median'
        limit_distance = 4
        include_query_terms = True
        return ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms
    

    def __get_ranking_parameters_config_02(self):
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_ranking_parameters_default_config()
        include_query_terms = False
        return ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms
    

    def __get_ranking_parameters_config_03(self):
        ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms = self.__get_ranking_parameters_default_config()
        summarize = 'median'
        return ranking_weight_type, lema, stem, summarize, limit_distance, include_query_terms
    

    def __initialize_binary_expression_tree_with_text_transformations(self, query: str) -> BinaryExpressionTree:
        # Load stopwords from JSON file
        stop_words = DataUtils.load_stopwords()

        # Initialize binary expression tree query
        lema = True
        stem = False

        b_expr_tree = BinaryExpressionTree(query)
        b_expr_tree.do_text_transformations_to_query_terms(stop_words, lema, stem)
        return b_expr_tree
    
    
    def __initialize_graph_1_and_2(self) -> tuple[VicinityGraph, VicinityGraph]:
        # Initialize vicinity graphs
        g1 = VicinityGraph(subquery="")
        g2 = VicinityGraph(subquery="")

        # Build graph 1
        g1_node1 = VicinityNode(term='x', ponderation=2.0, distance=np.mean([3,4]))
        g1_node2 = VicinityNode(term='y', ponderation=3.0, distance=np.mean([1,3,5]))
        g1.add_node(g1_node1)
        g1.add_node(g1_node2)

        # Build graph 2
        g2_node1 = VicinityNode(term='x', ponderation=2.0, distance=np.mean([1,2]))
        g2_node2 = VicinityNode(term='y', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2_node3 = VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2.add_node(g2_node1)
        g2.add_node(g2_node2)
        g2.add_node(g2_node3)
        
        return g1, g2
    
    
    def __get_results_from_generate_nodes_in_all_leaf_graphs(self, ranking: Ranking) -> tuple[dict, dict, dict]:
        ranking.calculate_vicinity_matrix_of_sentences_by_doc()
        
        # Generate nodes in all graphs in leaf nodes of the query expression tree
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.generate_nodes_in_all_leaf_graphs()

        # Get vicinity graphs from the sentence, as dicts
        query_terms = sentence.get_query_tree().get_query_terms_str_list_with_underscores()
        result1 = sentence.get_graph_by_subquery(query_terms[0]).get_graph_as_dict()
        result2 = sentence.get_graph_by_subquery(query_terms[1]).get_graph_as_dict()
        result3 = sentence.get_graph_by_subquery(query_terms[2]).get_graph_as_dict()
        
        return result1, result2, result3



if __name__ == '__main__':
    unittest.main()