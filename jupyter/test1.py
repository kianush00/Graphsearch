import unittest
import srex_new_classes as srex
import numpy as np
from collections import defaultdict

class TestSREX(unittest.TestCase):
    
    def test_union_between_graphs(self):
        g1 = srex.VicinityGraph(subquery="")
        g2 = srex.VicinityGraph(subquery="")
        r = srex.VicinityGraph(subquery="")

        g1_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([3,4]))
        g1_node2 = srex.VicinityNode(term='y', ponderation=3.0, distance=np.mean([1,3,5]))
        g1.add_node(g1_node1)
        g1.add_node(g1_node2)

        g2_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([1,2]))
        g2_node2 = srex.VicinityNode(term='y', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2_node3 = srex.VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2.add_node(g2_node1)
        g2.add_node(g2_node2)
        g2.add_node(g2_node3)

        r_node1 = srex.VicinityNode(term='x', ponderation=4.0, distance=np.mean([1, 2, 3, 4]))
        r_node2 = srex.VicinityNode(term='y', ponderation=7.0, distance=np.mean([11, 23, 52, 3, 1, 3, 5]))
        r_node3 = srex.VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        r.add_node(r_node1)
        r.add_node(r_node2)
        r.add_node(r_node3)

        self.assertDictEqual(g1.get_union_to_graph(g2).get_graph_as_dict(), r.get_graph_as_dict())
    

    def test_intersection_between_graphs(self):
        g1 = srex.VicinityGraph(subquery="")
        g2 = srex.VicinityGraph(subquery="")
        r = srex.VicinityGraph(subquery="")

        g1_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([3,4]))
        g1_node2 = srex.VicinityNode(term='y', ponderation=3.0, distance=np.mean([1,3,5]))
        g1.add_node(g1_node1)
        g1.add_node(g1_node2)

        g2_node1 = srex.VicinityNode(term='x', ponderation=2.0, distance=np.mean([1,2]))
        g2_node2 = srex.VicinityNode(term='y', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2_node3 = srex.VicinityNode(term='z', ponderation=4.0, distance=np.mean([11,23,52,3]))
        g2.add_node(g2_node1)
        g2.add_node(g2_node2)
        g2.add_node(g2_node3)

        r_node1 = srex.VicinityNode(term='x', ponderation=4.0, distance=np.mean([1, 2, 3, 4]))
        r_node2 = srex.VicinityNode(term='y', ponderation=7.0, distance=np.mean([11, 23, 52, 3, 1, 3, 5]))
        r.add_node(r_node1)
        r.add_node(r_node2)

        self.assertDictEqual(g1.get_intersection_to_graph(g2).get_graph_as_dict(), r.get_graph_as_dict())
        

    def test_calculate_ponderation_of_distances_between_term_positions(self):
        s = srex.Sentence(raw_text="", query=srex.BinaryExpressionTree(raw_query="test"))
        
        tp1 = [0,2,4,6]
        tp2 = [1,3,5,7]
        limit_distance = 7
        result = [7.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0]

        self.assertCountEqual(s.calculate_ponderation_of_distances_between_term_positions(tp1, tp2, limit_distance), result)
        

    def test_term_positions_dict(self):
        ranking = srex.Ranking(query_text="test")
        ranking.calculate_article_dictionaries_list([{}])

        text = 'hierarchical simknn driven dual index driven hierarchical network adopt hierarchical rtree driven store adopt topology driven hierarchical road network adopt network hierarchical topology'
        dict_result = {'hierarchical': [0, 6, 9, 16, 21], 'simknn': [1], 'driven': [2, 5, 11, 15], 'dual': [3], 'index': [4], 'network': [7, 18, 20], 'adopt': [8, 13, 19], 'rtree': [10], 'store': [12], 'topology': [14, 22], 'road': [17]}
        result = defaultdict(list, dict_result)
        
        self.assertEqual(ranking.get_documents()[0].get_sentences()[0].get_term_positions_dict(text), result)
    

    def test_get_transformed_text(self):
        text = "The European languages are members of the same family." + "Their separate existence is a myth. For science, music, sport, etc, Europe uses the same vocabulary. " + "The languages only differ in their grammar, their pronunciation and their most common words." + "Everyone realizes why a new common language would be desirable: one could refuse to pay expensive translators." + "To achieve this, it would be necessary to have uniform grammar, pronunciation and more common words." + "If several languages coalesce, the grammar of the resulting language is more simple and regular than that of the individual languages." + "The new common language will be more simple and regular than the existing European languages." + "It will be as simple as Occidental; in fact, it will be Occidental." + "To an English person, it will seem like simplified English, as a skeptical Cambridge friend of mine told me what Occidental is." + "The European languages are members of the same family. " + "Their separate existence is a myth. For science, music, sport, etc, Europe uses the same vocabulary. " + "The languages only differ in their grammar, their pronunciation and their most common words. " + "Everyone realizes why a new common language would be desirable: one could refuse to pay expensive translators."
        stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        lema = True
        stem = False
        result = "european language member separate existence myth science music sport etc europe us vocabulary language differ grammar pronunciation common realizes new common language desirable one refuse pay expensive achieve necessary uniform grammar pronunciation common several language coalesce grammar resulting language simple regular individual new common language simple regular existing european simple occidental fact english person seem like simplified english skeptical cambridge friend mine told occidental european language member family separate existence myth science music sport etc europe us vocabulary language differ grammar pronunciation common word everyone realizes new common language desirable one refuse pay expensive translator"

        self.assertEqual(srex.TextUtils.get_transformed_text(text, stop_words, lema, stem), result)
    

    def test_get_transformed_text_if_it_has_underscores(self):
        text_with_underscores = "internet_of_things"
        stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        lema = True
        stem = False
        result = "internet_thing"

        self.assertEqual(srex.TextUtils.get_transformed_text_if_it_has_underscores(text_with_underscores, stop_words, lema, stem), result)
    

    def test_cosine_similarity(self):
        stop_words               = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        query                    = 'network'
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        limit_distance           = 4 
        include_reference_terms  = True

        text1 = 'perro hierarchical SIMkNN dual-index driven, children network we adopt hierarchical gato R-tree to store the topology of the road network and gato hierarchical topology grid model to manage the moving objects in non-uniform distribution perro'
        text2 = 'perro hierarchical SIMkNN dual-index driven, house network car adopt hierarchical gato R-tree to store the topology of the road network and gato hierarchical topology grid model to manage the moving objects in non-uniform distribution perro'
        text3 = 'perro hierarchical SIMkNN dual-index driven, echeverria network patricio adopt hierarchical gato R-tree to galeas the topology of the road network abel gato hierarchical topology grid model temuco manage galeas moving objects abel non-uniform patricio perro'
        text4 = 'ant hierarchical SIMkNN dual-index driven, echeverria network rain adopt hierarchical rain R-tree to galeas the topology of the road network abel gato hierarchical topology grid model rain manage galeas moving objects abel non-uniform patricio ant'

        list_of_articles_dicts = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list(list_of_articles_dicts)
        ranking.generate_all_graphs(limit_distance=limit_distance, include_query_terms=include_reference_terms, summarize=summarize)

        g1 = ranking.get_document_by_ranking_position(1).get_graph()
        g2 = ranking.get_document_by_ranking_position(2).get_graph()
        g3 = ranking.get_document_by_ranking_position(3).get_graph()
        g4 = ranking.get_document_by_ranking_position(4).get_graph()

        self.assertAlmostEqual(g1.get_cosine_similarity(g1), 1.0, delta=1e-13)
        self.assertAlmostEqual(g1.get_cosine_similarity(g2), 0.9671707945560485, delta=1e-13)
        self.assertAlmostEqual(g1.get_cosine_similarity(g3), 0.6557450996618941, delta=1e-13)
        self.assertAlmostEqual(g1.get_cosine_similarity(g4), 0.5949574537482146, delta=1e-13)
    

    def test_get_terms_from_nodes(self):
        stop_words               = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        query                    = 'network'
        ranking_weight_type      = 'linear' # it can be: 'none', 'linear' or 'inverse'
        lema                     = True
        stem                     = False
        summarize                = 'mean'   # it can be: 'mean' or 'median'
        limit_distance           = 4 
        include_reference_terms  = True

        text1 = 'perro hierarchical SIMkNN dual-index driven, children network we adopt hierarchical gato R-tree to store the topology of the road network and gato hierarchical topology grid model to manage the moving objects in non-uniform distribution perro'
        text2 = 'perro hierarchical SIMkNN dual-index driven, house network car adopt hierarchical gato R-tree to store the topology of the road network and gato hierarchical topology grid model to manage the moving objects in non-uniform distribution perro'
        text3 = 'perro hierarchical SIMkNN dual-index driven, echeverria network patricio adopt hierarchical gato R-tree to galeas the topology of the road network abel gato hierarchical topology grid model temuco manage galeas moving objects abel non-uniform patricio perro'
        text4 = 'ant hierarchical SIMkNN dual-index driven, echeverria network rain adopt hierarchical rain R-tree to galeas the topology of the road network abel gato hierarchical topology grid model rain manage galeas moving objects abel non-uniform patricio ant'

        list_of_articles_dicts = [{'abstract': text1}, 
                                {'abstract': text2}, 
                                {'abstract': text3}, 
                                {'abstract': text4}]
        
        ranking = srex.Ranking(query, ranking_weight_type=ranking_weight_type, stop_words=stop_words, lemmatization=lema, stemming=stem)
        ranking.calculate_article_dictionaries_list(list_of_articles_dicts)
        ranking.generate_all_graphs(limit_distance=limit_distance, include_query_terms=include_reference_terms, summarize=summarize)

        g1 = ranking.get_document_by_ranking_position(1).get_graph()
        g2 = ranking.get_document_by_ranking_position(2).get_graph()
        g3 = ranking.get_document_by_ranking_position(3).get_graph()
        g4 = ranking.get_document_by_ranking_position(4).get_graph()

        result1 = ['hierarchical', 'adopt', 'grid', 'driven', 'gato', 'road', 'topology', 'store', 'simknn', 'child']
        result2 = ['hierarchical', 'car', 'adopt', 'grid', 'driven', 'gato', 'road', 'topology', 'house', 'store', 'simknn']
        result3 = ['hierarchical', 'adopt', 'driven', 'gato', 'road', 'galea', 'topology', 'echeverria', 'patricio', 'abel', 'simknn']
        result4 = ['hierarchical', 'adopt', 'driven', 'road', 'gato', 'galea', 'topology', 'echeverria', 'abel', 'simknn', 'rain']

        self.assertCountEqual(g1.get_terms_from_nodes(), result1)
        self.assertCountEqual(g2.get_terms_from_nodes(), result2)
        self.assertCountEqual(g3.get_terms_from_nodes(), result3)
        self.assertCountEqual(g4.get_terms_from_nodes(), result4)


if __name__ == '__main__':
    unittest.main()