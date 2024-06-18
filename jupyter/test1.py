import unittest
import srex_new_classes as srex
import numpy as np

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

        self.assertEqual(g1.get_union_to_graph(g2).get_graph_as_dict(), r.get_graph_as_dict())
    
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

        self.assertEqual(g1.get_intersection_to_graph(g2).get_graph_as_dict(), r.get_graph_as_dict())
        
    def test_calculate_termpositions_distances(self):
        tp1 = [0,2,4,6]
        tp2 = [1,3,5,7]
        limit_distance = 7
        result = [7.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0]

        s = srex.Sentence(raw_text="", query=srex.BinaryExpressionTree(raw_query="test"), weight=1.0)
        s.get_query_tree().initialize_graph_for_each_node(limit_distance=limit_distance)

        self.assertCountEqual(s.calculate_ponderation_of_distances_between_term_positions(tp1, tp2), result)
        
    # def test_term_positions_dict(self):
    #     ranking = srex.Ranking(query_text="test")
    #     doc_dict = {'abstract': 'hierarchical simk driven dual index driven hierarchical network adopt hierarchical rtree driven store adopt topology driven hierarchical road network adopt network hierarchical topology'}
    #     ranking.calculate_article_dictionaries_list([doc_dict])
    #     ranking.get_documents()[0].get_sentences()[0].do_term_positions_dict()

    #     print(dict(ranking.get_documents()[0].get_sentences()[0].get_term_positions_dict()))
        
    #     result = {'hierarchical': [0, 6, 9, 16, 21], 'simk': [1], 'driven': [2, 5, 11, 15], 'dual': [3], 'index': [4], 'network': [7, 18, 20], 'adopt': [8, 13, 19], 'rtree': [10], 'store': [12], 'topology': [14, 22], 'road': [17]}
    #     self.assertEqual(ranking.get_documents()[0].get_sentences()[0].get_term_positions_dict(), result)


if __name__ == '__main__':
    unittest.main()