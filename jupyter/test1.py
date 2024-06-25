import unittest
import srex_new_classes as srex
import numpy as np
from collections import defaultdict
import json

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
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
        text = test_data.get('text5')
        expected_result = test_data.get('text5_remove_special_characters')
        
        # Tokenize the text
        tokens = srex.nltk.word_tokenize(text.lower())
        
        # Remove special characters
        result = srex.TextUtils.remove_special_characters(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_remove_stopwords(self):
        # Load test text data
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
        text = test_data.get('text5')
        expected_result = test_data.get('text5_remove_stopwords')
        
        # Load stopwords
        with open('jupyter/json_data/stopwords_data.json') as f:
            stopwords_data = json.load(f)
        stop_words = stopwords_data.get('words')
        
        # Tokenize the text
        tokens = srex.nltk.word_tokenize(text.lower())
        
        # Remove stopwords
        result = srex.TextUtils.remove_stopwords(tokens, stop_words)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)

    
    def test_do_lemmatization(self):
        # Load test text data
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
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
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
        text = test_data.get('text5')
        expected_result = test_data.get('text5_do_stemming')
        
        # Tokenize the text
        tokens = srex.nltk.word_tokenize(text.lower())
        
        # Perform stemming
        result = srex.TextUtils.do_stemming(tokens)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)


    def test_get_transformed_text_with_lema(self):
        # Load test data and expected result from JSON file
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
        text = test_data.get('text5')
        expected_result = test_data.get('text5_transformed_text_with_lema')
        
        # Load stopwords from JSON file
        with open('jupyter/json_data/stopwords_data.json') as f:
            stopwords_data = json.load(f)
        
        stop_words = stopwords_data.get('words')
        
        # Set lema and stem flags
        lema = True
        stem = False
        
        # Get transformed text
        result = srex.TextUtils.get_transformed_text(text, stop_words, lema, stem)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_get_transformed_text_if_it_has_underscores(self):
        text_with_underscores = "internet_of_things"
        expected_result = "internet_thing"
        
        # Load stopwords from JSON file
        with open('jupyter/json_data/stopwords_data.json') as f:
            stopwords_data = json.load(f)
        stop_words = stopwords_data.get('words')
        
        # Set lema and stem flags
        lema = True
        stem = False
        
        # Get transformed text
        result = srex.TextUtils.get_transformed_text_if_it_has_underscores(text_with_underscores, stop_words, lema, stem)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
        

    def test_term_positions_dict(self):
        # Initialize Ranking object
        ranking = srex.Ranking(query_text="test")
        ranking.calculate_article_dictionaries_list([{'title': 'test'}])

        # Load test data from JSON file
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
        text = test_data.get('text6')

        # Expected term positions dictionary
        dict_result = {
            'hierarchical': [0, 6, 9, 16, 21], 'simknn': [1], 'driven': [2, 5, 11, 15], 'dual': [3], 'index': [4], 
            'network': [7, 18, 20], 'adopt': [8, 13, 19], 'rtree': [10], 'store': [12], 'topology': [14, 22], 'road': [17]
        }

        # Get term positions dictionary from ranking object
        result = ranking.get_documents()[0].get_sentences()[0].get_term_positions_dict(text)
        
        # Create expected result as defaultdict
        expected_result = defaultdict(list, dict_result)
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_query_term_positions_dict(self):
        # Initialize Ranking object
        ranking = srex.Ranking(query_text="test")
        ranking.calculate_article_dictionaries_list([{'title': 'test'}])

        query_terms = ["driven", "adopt", "store"]

        # Term positions dictionary
        term_positions_dict = {
            'hierarchical': [0, 6, 9, 16, 21], 'simknn': [1], 'driven': [2, 5, 11, 15], 'dual': [3], 'index': [4], 
            'network': [7, 18, 20], 'adopt': [8, 13, 19], 'rtree': [10], 'store': [12], 'topology': [14, 22], 'road': [17]
        }

        # Get query term positions dictionary from ranking object
        result = ranking.get_documents()[0].get_sentences()[0].get_query_term_positions_dict(term_positions_dict, query_terms)
        
        # Create expected result
        expected_result = {'driven': [2, 5, 11, 15], 'adopt': [8, 13, 19], 'store': [12]}
        
        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_calculate_vicinity_matrix_with_include_query_terms(self):
        # Load stopwords
        with open('jupyter/json_data/stopwords_data.json') as f:
            stopwords_data = json.load(f)
        stop_words = stopwords_data.get('words')

        # Load test data from JSON file
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
        text = test_data.get('text6')
        expected_result = test_data.get('text6_calculate_vicinity_matrix_with_include_query_terms')
        article_dict = {'abstract': text}

        # Initialize Ranking object with include_query_terms as true
        query                    = 'driven OR adopt OR store'
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        limit_distance           = 4 
        include_query_terms      = True

        # Initialize Ranking object
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
        # Load stopwords
        with open('jupyter/json_data/stopwords_data.json') as f:
            stopwords_data = json.load(f)
        stop_words = stopwords_data.get('words')

        # Load test data from JSON file
        with open('jupyter/json_data/test_data.json') as f:
            test_data = json.load(f)
        text = test_data.get('text6')
        expected_result = test_data.get('text6_calculate_vicinity_matrix_without_include_query_terms')
        article_dict = {'abstract': text}

        # Initialize Ranking object with include_query_terms as false
        query                    = 'driven OR adopt OR store'
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        limit_distance           = 4 
        include_query_terms      = False

        # Initialize Ranking object
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list([article_dict])
        ranking.initialize_graphs_for_all_trees(limit_distance=limit_distance, include_query_terms=include_query_terms)
        
        #Calculate the vicinity matrix for the sentence
        sentence = ranking.get_document_by_ranking_position(1).get_sentence_by_position_in_doc(0)
        sentence.calculate_vicinity_matrix()
        result = sentence.get_vicinity_matrix()

        # Assert the result matches the expected output
        self.assertEqual(result, expected_result)
    

    def test_cosine_similarity(self):
        # Load stopwords
        with open('jupyter/json_data/stopwords_data.json') as f:
            stopwords_data = json.load(f)
        stop_words = stopwords_data.get('words')

        # Load test data from JSON file
        with open('jupyter/json_data/test_data.json') as f:
            text_json = json.load(f)
        text1 = text_json.get('text1')
        text2 = text_json.get('text2')
        text3 = text_json.get('text3')
        text4 = text_json.get('text4')

        list_of_articles_dicts = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        # Initialize Ranking object
        query                    = 'network'
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        limit_distance           = 4 
        include_query_terms  = True
        
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
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
        self.assertAlmostEqual(result1, expected_result1, delta=1e-13)
        self.assertAlmostEqual(result2, expected_result2, delta=1e-13)
        self.assertAlmostEqual(result3, expected_result3, delta=1e-13)
        self.assertAlmostEqual(result4, expected_result4, delta=1e-13)
    

    def test_get_terms_from_nodes(self):
        # Load stopwords
        with open('jupyter/json_data/stopwords_data.json') as f:
            stopwords_data = json.load(f)
        stop_words = stopwords_data.get('words')

        # Load test data from JSON file
        with open('jupyter/json_data/test_data.json') as f:
            text_json = json.load(f)
        text1 = text_json.get('text1')
        text2 = text_json.get('text2')
        text3 = text_json.get('text3')
        text4 = text_json.get('text4')
        expected_result1 = text_json.get('text1_terms_from_nodes')
        expected_result2 = text_json.get('text2_terms_from_nodes')
        expected_result3 = text_json.get('text3_terms_from_nodes')
        expected_result4 = text_json.get('text4_terms_from_nodes')

        list_of_articles_dicts = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        # Initialize Ranking object and initialize all graphs
        query                    = 'network'
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        limit_distance           = 4 
        include_query_terms  = True
        
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
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


if __name__ == '__main__':
    unittest.main()