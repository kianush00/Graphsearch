import jupyter.old.srex as srex
import operator
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

class Ranking:
    
    def __init__(self, url):
        self.__url = url
        self.__document_ranking = []

        

class Document:
    
    def __init__(self, docId, url, ranking, title, document_content, reference_term, limit_distance, sumarize, include_reference_term, nr_of_graph_terms, stop_words, lema, stem):
        
        self.__docId = docId
        self.__url = url
        self.__ranking = ranking
        self.__title = title
        self.__graphs_array = []

        # Configuration variables
        self.__reference_term = reference_term
        self.__limit_distance = limit_distance
        self.__sumarize = sumarize
        self.__include_reference_term = include_reference_term
        self.__nr_of_graph_terms = nr_of_graph_terms
        self.__stop_words = stop_words
        self.__lema = lema
        self.__stem = stem
        
        
        # Split the document in paragraphs
        text_paragraphs_array = self.__splitContent(document_content)
        
        # Clean their content (stopwords, steming, etc.)
        cleaned_text_paragraphs_array = self.__cleanParagraphs(text_paragraphs_array)
        
        # Create the graphs objects
        self.__createGraphs(cleaned_text_paragraphs_array)
    
    
    # Divide de document content into an array of paragraphs     
    def __splitContent(self, content, min_paragraph_size=2):
        text_paragraphs_array_tmp = content.split('.')
        print("text_paragraphs_array_tmp: "+str(len(text_paragraphs_array_tmp)))
        i=0
        for p in text_paragraphs_array_tmp:
            i=i+1
            print(str(i) + " : " + p)
        text_paragraphs_array_filtered = list(filter(lambda paragraph: len(paragraph) > min_paragraph_size, text_paragraphs_array_tmp))
        print("text_paragraphs_array_filtered: "+str(len(text_paragraphs_array_filtered)))
        i=0
        for q in text_paragraphs_array_filtered:
            i=i+1
            print(str(i) + " : " + q)
        return text_paragraphs_array_filtered
    

    # Apply text processing methods to gthe paragraph array to clean its content 
    def __cleanParagraphs(self, text_paragraphs_array):
        cleaned_text_paragraphs_array = list(map(lambda x: srex.text_transformations(x, self.__stop_words, self.__lema, self.__stem), text_paragraphs_array))
        return cleaned_text_paragraphs_array
       
        
    # Create the array of Paragraph objects
    def __createGraphs(self, cleaned_text_paragraphs_array):
        # Create an arrar of Paragraph objects   
        for cleaned_text_paragraph in cleaned_text_paragraphs_array:
            #print("cleaned_text_paragraph : " + cleaned_text_paragraph)
            graph = Graph(self.__docId, cleaned_text_paragraph, self.__reference_term, self.__limit_distance, self.__sumarize, self.__include_reference_term, self.__nr_of_graph_terms, self.__stop_words, self.__lema, self.__stem)
            graph.calculateGraphNodes()
            #print(graph)
            self.__graphs_array.append(graph)
    
    # Return the graph objects associated to each paragraphcument
    def getGraphs(self):
        return self.__graphs_array
    
    
    # Calculate the Document GraphDictionary with the terms frecuency 
    # and all term-distances to the reference term, considering all Paragraphs in the document
    # def createDocumentGraphDictionary(self):
        documentGraphDictionary = srex.get_unique_graph_dictionary(self.__paragraphs_array)

        
        
        
class Graph:
        
    def __init__(self, docId, paragraph_text, reference_term, limit_distance, sumarize, include_reference_term, nr_of_graph_terms, stop_words, lema, stem):

        self.__docId = docId
        self.__paragraph_text = paragraph_text
        self.nodes = {}

        self.reference_term = reference_term
        self.__limit_distance = limit_distance
        self.__sumarize = sumarize
        self.__include_reference_term = include_reference_term
        self.__nr_of_graph_terms = nr_of_graph_terms
        self.__stop_words = stop_words
        self.__lema = lema
        self.__stem = stem
        
        
    def getParagraph_text(self):
        return self.__paragraph_text

    
    # Clean the text of the paragraph using some text processing algorithms 
    def cleanContent(self):
        self.__paragraph_text = srex.text_transformations(self.__paragraph_text, self.__stop_words, self.__lema, self.__stem)   
        
        
    # Calculate Nodes
    def calculateGraphNodes(self):
        # Get the term positions dictionary
        terms_positions_dict = srex.get_term_positions_dict(self.__paragraph_text)
        
        # Get the reference_term's vecinity dictionary
        document_term_vecinity_dict = srex.get_document_term_vecinity_dict(terms_positions_dict, self.reference_term, self.__limit_distance, self.__sumarize, self.__include_reference_term)
        
        # Get the term frequency dictionary
        terms_freq_dict = {k: len(v) for k, v in document_term_vecinity_dict.items()}
        
        # Sort the most frequently terms
        sorted_terms_freq_dict = sorted(terms_freq_dict.items(), key=operator.itemgetter(1), reverse=True)
        
        # Filter the top most frequently terms for the graph
        first_sorted_terms_freq_dict = {k: v for k, v in list(sorted_terms_freq_dict)[:self.__nr_of_graph_terms]}
        
        # Get graph dictionary
        self.nodes = {k: {'frequency':terms_freq_dict[k], 'distances':document_term_vecinity_dict[k]} for k in first_sorted_terms_freq_dict.keys()}
        
        
    # Calculate the base terms from two graphs
    def __getVectorBase(self, otherGraph):
        vectorSpace=[]
        s1 = set(self.nodes.keys())
        s2 = set(otherGraph.nodes.keys())
        vectorBase = sorted(list(s1 | s2)) # Calculate de total terms
        return vectorBase

    # Integrate the frequency/distance lists to build the graph object
    def __integrate(self, frequency, distance):
        return (frequency[0], {'frequency':frequency[1], 'distance':distance[1]})
    
    
    # Return the graph with a summarized distance values
    def getSummarizedGraph(self, summarize='median', normalize_frequency_range=False, normalize_distance_range=False):
        # Calculate a term distance metric for all nodes
        if (summarize =='median'):
            summarized_graph = {k: {'frequency':self.nodes[k]['frequency'], 'distance':srex.np.median(self.nodes[k]['distances'])} for k in self.nodes.keys()}
        elif (summarize == 'mean'):
            summarized_graph = {k: {'frequency':self.nodes[k]['frequency'], 'distance':srex.np.mean(self.nodes[k]['distances'])} for k in self.nodes.keys()}
        
        if((normalize_frequency_range != True) | (normalize_distance_range != False)):
            summarized_graph_distance = {k: v['distance'] for k, v in summarized_graph.items()}
            summarized_graph_frequency = {k: v['frequency'] for k, v in summarized_graph.items()}
            
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
    
    
    
    # Plot the graph
    def plotGraph(self, summarize='median', normalized=False, node_size='0.7', node_color='deepskyblue'):
        sumarized_graph = self.getSummarizedGraph(summarize, normalized)
        return srex.getGraphViz(self.reference_term, sumarized_graph, node_size, node_color)    
    
    
    
    # Add terms from otherGraph to self.nodes
    def merge(self, otherGraph):
        if(otherGraph.reference_term == self.reference_term):
            srex.merge_graph_dictionaries(otherGraph.nodes, self.nodes)
        else:
            print("Merge ERROR: Graphs have different reference terms.")
        return self
            
    

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
        for term in vectorBase: # Generate de vector space for both attributes (frequency and distance)
            if (term in self_sum):
                u.append(self_sum[term]['frequency'])
                u.append(self_sum[term]['distance'])
            else:
                u.append(0) # frequency value equal to cero
                u.append(0) # distance value equal to cero
            if (term in otherGraph_sum):
                v.append(otherGraph_sum[term]['frequency'])
                v.append(otherGraph_sum[term]['distance'])
            else:
                v.append(0) # frequency value equal to cero
                v.append(0) # distance value equal to cero

        # Calculate the cosine of the angle between the vectors
        cosine_of_angle = dot(u,v)/norm(u)/norm(v)
        
        return cosine_of_angle
    
    
    # Normalize the graphs values using a user defined range
    # The range should be (lower_bound, upper_bound)     
    def normalize(self,frequency_range, distance_range):
        print("normalize(self, frequency_range, distance_range) --- not yet implemented")
        a1 = dictio[max(dictio, key=dictio.get)]
        c1 = dictio[min(dictio, key=dictio.get)]
        b1 = frequency_range[1]
        d1 = frequency_range[0]
        
        if((a1 - c1)>0):
            m1 = (b1 - d1) / (a1 - c1)
        else:
            m1 = (b1 - d1) # term frequency dictionary have only sigle words (frequency=1)    
            
        a2 = dictio[max(dictio, key=dictio.get)]
        c2 = dictio[min(dictio, key=dictio.get)]
        b2 = distance_range[1]
        d2 = distance_range[0]
        
        if((a2 - c2)>0):
            m2 = (b2 - d2) / (a2 - c2)
        else:
            m2 = (b2 - d2) # term frequency dictionary have only sigle words (frequency=1)
        
        dictio.update((k, (m*(dictio[k]-c)+d)) for k in dictio.keys())

    
    # Remove irrelevant terms from the graph
    def pruneTerms(self,nrNodes):
        print("pruneTerms(self,nrNodes) --- not yet implemented") 
    
    
    def __str__(self):
        graph_str = 'REF-TERM:' + self.reference_term + '\nNODES:' + str(self.nodes)
        return graph_str

