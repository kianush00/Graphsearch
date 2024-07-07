from collections import deque 
import re
import copy
from functools import reduce
from utils.text_utils import TextUtils
from models.srex.vicinity_graph import VicinityGraph




class BinaryTreeNode:
    def __init__(self, value: str):
        self.value = value
        self.graph: VicinityGraph | None = None
        self.left: BinaryTreeNode | None = None
        self.right: BinaryTreeNode | None = None
    

    def is_leaf(self) -> bool:
        is_leaf = not self.left and not self.right
        return is_leaf
    

    def get_values_from_leaves(self) -> list[str]:
        if self.is_leaf():
            return [self.value]
        return self.left.get_values_from_leaves() + self.right.get_values_from_leaves()
    

    def get_leaves(self) -> list['BinaryTreeNode']:
        if self.is_leaf():
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()
    

    def get_graph_from_subtree_by_subquery(self, query: str) -> VicinityGraph | None:
        if self.graph and self.graph.subquery == query:
            return self.graph
            
        # Check left subtree
        if self.left:
            left_candidate = self.left.get_graph_from_subtree_by_subquery(query)
            if left_candidate:
                return left_candidate

        # Check right subtree
        if self.right:
            right_candidate = self.right.get_graph_from_subtree_by_subquery(query)
            if right_candidate:
                return right_candidate

        return None
    

    def get_union_to_subtree(self, 
            external_subtree: 'BinaryTreeNode'
            ) -> 'BinaryTreeNode':
        """
        Gets the union between an external subtree and the own subtree, then obtains 
        a new subtree. The merging process involves iterating through the nodes of both 
        subtrees, calculating  the sum of weights and the average distances between 
        each one, that is, the union between each graph (tree node) of both subtrees.
        
        Parameters
        ----------
        external_subtree : BinaryTreeNode
            The external subtree to be united

        Returns
        -------
        copy_subtree : BinaryTreeNode
            The union between copy of the subtree itself and an external subtree
        """
        copy_subtree = copy.deepcopy(self)
        copy_subtree.__do_union_between_copy_self_and_subtree(external_subtree)
        return copy_subtree
    

    def do_graph_operation_from_subtrees(self) -> None:
        if not self.is_leaf():
            #Validate if the tree has no empty vicinity graphs in leaves
            try:
                self.__set_and_get_graph_operated_from_subtrees()
            except Exception as e:
                print('Error at do_graph_operation_from_subtrees(): ' + repr(e))
    

    def __str__(self) -> str:
        return ''.join(self.__inorder_traversal())
    

    def tree_str(self) -> str:
        return self.__generate_tree_str()
    

    def __do_union_between_copy_self_and_subtree(self, 
            external_peer_node: 'BinaryTreeNode'
            ) -> None:
        """
        Unites an external subtree with the own copy subtree and modifies 
        the own subtree graphs. This method should only be used by the 
        copy from the original BinaryTreeNode object.
        
        Parameters
        ----------
        external_peer_node : BinaryTreeNode
            The external peer node to be united
        """
        #Checks if both nodes have a non-null graph attribute, to then join them and set 
        #them in the copy attribute
        if self.graph and external_peer_node.graph:
            current_node_united_graph = self.graph.get_union_to_graph(external_peer_node.graph)
            self.graph = current_node_united_graph
        
        #Checks if both nodes have a left node, to continue the recursion
        if self.left and external_peer_node.left:
            self.left.__do_union_between_copy_self_and_subtree(external_peer_node.left)
        
        #Checks if both nodes have a right node, to continue the recursion
        if self.right and external_peer_node.right:
            self.right.__do_union_between_copy_self_and_subtree(external_peer_node.right)
    

    def __set_and_get_graph_operated_from_subtrees(self) -> VicinityGraph:
        graph = None
        left_graph = None
        right_graph = None
        
        if self.left.is_leaf():
            if self.left.graph:
                left_graph = self.left.graph
            else:
                raise Exception('Null vicinity graphs in leaves')
        else:
            left_graph = self.left.__set_and_get_graph_operated_from_subtrees()
        
        if self.right.is_leaf():
            if self.right.graph:
                right_graph = self.right.graph
            else:
                raise Exception('Null vicinity graphs in leaves')
        else:
            right_graph = self.right.__set_and_get_graph_operated_from_subtrees()

        if left_graph and right_graph:
            if left_graph.subquery == right_graph.subquery:
                graph = left_graph
            elif self.value == 'AND':
                graph = left_graph.get_intersection_to_graph(right_graph)
            elif self.value == 'OR':
                graph = left_graph.get_union_to_graph(right_graph)
            
            #The new graph inherits the same subquery from the self graph
            if self.graph:
                graph.subquery = self.graph.subquery

            self.graph = graph
            return graph
    

    def __inorder_traversal(self, level: int = 0) -> list[str]:
        result = []

        # Traverse left subtree
        if self.left:
            # Add opening parenthesis if not at root level
            if level > 0: result.append('(')
            # Recursively traverse left subtree and increase level
            result.extend(self.left.__inorder_traversal(level + 1))

        # Append current node value with spaces if it's 'AND' or 'OR'
        if self.value in ['AND', 'OR']:    
            result.append(f' {self.value} ')
        else:
            result.append(self.value)

        # Traverse right subtree
        if self.right:
            # Recursively traverse right subtree and increase level
            result.extend(self.right.__inorder_traversal(level + 1))
            # Add closing parenthesis if not at root level
            if level > 0: result.append(')')
        return result
    

    def __generate_tree_str(self, level: int = 0) -> str:
        output = "\t" * level + self.value + "\n"
        if self.left:
            output += self.left.__generate_tree_str(level + 1)
        if self.right:
            output += self.right.__generate_tree_str(level + 1)
        return output



class BinaryExpressionTree:
    def __init__(self, raw_query: str):
        self.__raw_query = raw_query
        self.root = self.__get_tree_built()
    

    def get_raw_query(self) -> str:
        return self.__raw_query
    

    def get_query_terms_str_list_with_underscores(self) -> list[str]:
        return self.root.get_values_from_leaves()
    

    def get_query_terms_as_leaves(self) -> list[BinaryTreeNode]:
        return self.root.get_leaves()
    

    def get_graph(self) -> VicinityGraph | None:
        return self.root.graph
    

    def get_graph_by_subquery(self, query: str) -> VicinityGraph | None:
        graph = self.root.get_graph_from_subtree_by_subquery(query)
        if not graph:
            print('Could not find graph for subquery')
            return None
        return graph
    

    def get_union_of_trees(self, 
            query_trees_list: list['BinaryExpressionTree']
            ) -> 'BinaryExpressionTree':
        """
        Get the union between the given list of query trees.

        Parameters
        ----------
        query_trees_list : list[BinaryExpressionTree]
            The list of query trees to be united between

        Returns
        -------
        union_of_trees : BinaryExpressionTree
            The union between the query trees
        """
        if not query_trees_list:
            print("query_trees_list must not be empty")
            return

        # reduce() applies a function of two arguments cumulatively to the items of a sequence or 
        #iterable, from left to right, so as to reduce the iterable to a single value. 
        #For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5)
        union_of_trees = reduce((lambda tree1, tree2: tree1.get_union_to_tree(tree2)), query_trees_list)
        return union_of_trees
    

    def get_union_to_tree(self,
            external_tree: 'BinaryExpressionTree'
            ) -> 'BinaryExpressionTree':
        """
        Unites an external tree with the own tree and obtains a new tree.
        The merging process involves iterating through the graphs of both tree, calculating 
        the sum of weights and the average distances between each one, that is, the union
        between each graph (tree node) of both trees.
        
        Parameters
        ----------
        external_tree : BinaryExpressionTree
            The external tree to be united

        Returns
        -------
        copy_tree : BinaryExpressionTree
            The union between copy of the tree itself and an external tree
        """
        copy_tree = copy.deepcopy(self)
        
        #Get the deep union between the copy root node and the external root node
        copy_root = copy_tree.root.get_union_to_subtree(external_tree.root)
        copy_tree.root = copy_root
        return copy_tree
    

    def operate_graphs_from_leaves(self) -> None:
        self.root.do_graph_operation_from_subtrees()
    

    def do_text_transformations_to_query_terms(self, 
            stop_words_list: list[str] = [], 
            lema: bool = True, 
            stem: bool = False
            ) -> None:
        """
        Apply text transformations to the terms in the leaf nodes.

        Parameters
        ----------
        stop_words_list : list[str], optional
            List of stop words to be removed from the terms
        lema : bool, optional
            If True, lemmatization is applied
        stem : bool, optional
            If True, stemming is applied
        """
        def transform_node_if_leaf(node: BinaryTreeNode) -> None:
            if node.is_leaf():
                node.value = TextUtils.get_transformed_text_if_it_has_underscores(node.value, stop_words_list, 
                                                                                  lema, stem)
            else:
                transform_node_if_leaf(node.left)
                transform_node_if_leaf(node.right)

        transform_node_if_leaf(self.root)
    

    def initialize_graph_for_each_node(self, 
            nr_of_graph_terms: int = 5, 
            limit_distance: int = 4, 
            include_query_terms: bool = True,
            summarize: str = 'mean'
            ) -> None:
        """
        Initialize the graph associated to each node in the tree.

        Parameters
        ----------
        nr_of_graph_terms : int, optional
            Configured number of terms in the graph
        limit_distance : int, optional
            Maximal distance of terms used to calculate the vicinity
        include_query_terms : bool, optional
            If True, the query term is included in the vicinity
        summarize : str, optional
            Summarization type to operate distances in the vicinity matrix for each 
            sentence (it can be: mean or median)
        """
        def initialize_graph(node: BinaryTreeNode):
            node.graph = VicinityGraph(str(node), nr_of_graph_terms, limit_distance, 
                                       include_query_terms, summarize)
            if not node.is_leaf():
                initialize_graph(node.left)
                initialize_graph(node.right)
        
        initialize_graph(self.root)
    

    def remove_graphs_for_each_node(self) -> None:
        """
        Remove the graph associated to each node in the tree.
        """
        def remove_graph(node: BinaryTreeNode):
            node.graph = None
            if node.left:
                remove_graph(node.left)
            if node.right:
                remove_graph(node.right)
        
        remove_graph(self.root)


    def __str__(self) -> str:
        return str(self.root)
    

    def tree_str(self) -> str:
        return self.root.tree_str()
    
    
    def __get_tree_built(self):
        tokens = self.__separate_boolean_query(self.__raw_query)
        infix_tokens = self.__process_infix_tokens(tokens)
        postfix_tokens = self.__infix_to_postfix(infix_tokens)
        root_node = self.__construct_tree_from_postfix(postfix_tokens)
        return root_node
    

    def __separate_boolean_query(self, query: str) -> list[str]:
        # Define a regex pattern to match boolean operators, parentheses, colons and terms
        pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\(|\)|\w+|:)'
        
        # Find all matches using the pattern
        tokens = re.findall(pattern, query)
        
        return tokens
    

    def __process_infix_tokens(self, tokens: list[str]) -> list[str]:
        validation_tuple = self.__validate_boolean_expression(tokens)
        if validation_tuple[0]:
            tokens_deleted_not_expressions = self.__process_boolean_tokens_delete_not_expressions(tokens)
            tokens_deleted_colon_and_keys = self.__process_boolean_tokens_delete_colon_and_keys(tokens_deleted_not_expressions)
            infix_tokens = self.__join_term_tokens_by_underscores(tokens_deleted_colon_and_keys)
            return infix_tokens
        else:
            raise Exception(validation_tuple[1])
    

    def __infix_to_postfix(self, infix_tokens: list[str]) -> list[str]:
        # Operator precedence
        precedence = {'OR': 1, 'AND': 2}
        # Stack for operators
        operator_stack = deque()
        # List for expression in postfix
        postfix_list = []

        for token in infix_tokens:
            if token in precedence.keys():
                # If the token is an operator, handle precedence (operator_stack[0] is the top element of stack)
                while (operator_stack and operator_stack[0] != '(' and
                    precedence[operator_stack[0]] >= precedence[token]):
                    postfix_list.append(operator_stack.popleft())
                operator_stack.appendleft(token)
            elif token == '(':
                # If the token is '(', add it to the operator stack
                operator_stack.appendleft(token)
            elif token == ')':
                # If the token is ')', pop until find '(' (operator_stack[0] is the top element of stack)
                while operator_stack and operator_stack[0] != '(':
                    postfix_list.append(operator_stack.popleft())
                operator_stack.popleft()  # Pop '(' from stack
            else:
                # If the token is an operand (alphanumeric word or containing '_'), add it to postfix
                postfix_list.append(token)

        # Add any remaining operators on the stack to the postfix list
        while operator_stack:
            postfix_list.append(operator_stack.popleft())

        return postfix_list
    
    
    def __construct_tree_from_postfix(self, postfix_tokens: list[str]) -> BinaryTreeNode:
        stack = []
        for token in postfix_tokens:
            if token in ['AND', 'OR']:
                # If the token is an operator, create a node and link the subtrees
                node = BinaryTreeNode(token)
                node.right = stack.pop()
                node.left = stack.pop()
                stack.append(node)
            else:
                # If the token is an operand, create a node and add it to the stack
                node = BinaryTreeNode(token)
                stack.append(node)
        
        # The last element on the stack is the root of the tree
        return stack.pop()


    def __validate_boolean_expression(self, tokens: list[str]) -> tuple[bool, str]:
        # Check parenthesis syntax
        stack = []
        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                if not stack:
                    return False, "Unmatched closing parenthesis"
                stack.pop()
        if stack:
            return False, "Unmatched opening parenthesis"

        # Check AND OR operators
        operators = {'AND', 'OR'}
        length = len(tokens)
        
        for i, token in enumerate(tokens):
            if token in operators:
                if i == 0 or i == length - 1:
                    return False, f"Operator {token} cannot be at the start or end of the expression"
                if tokens[i-1] in operators or tokens[i+1] in operators:
                    return False, f"Operator {token} must be surrounded by valid terms"

        return True, "The expression is valid"


    def __process_boolean_tokens_delete_not_expressions(self, tokens: list[str]) -> list[str]:
        indices_to_exclude = []
        index = 0

        while index < len(tokens):
            next_index = index + 1
            if tokens[index] == 'NOT':
                stack = 0
                while next_index < len(tokens):
                    if tokens[next_index] == '(':
                        stack += 1
                    elif tokens[next_index] == ')':
                        if stack < 2: # if it is the last closing parenthesis, or there's no opening parenthesis between the 2 indices
                            if stack == 1:
                                next_index += 1     # removes the closing parenthesis if there was an opening parenthesis before
                            break
                        else:
                            stack -= 1
                    elif tokens[next_index] in ['AND', 'OR', 'NOT']:
                        if stack < 1:   # if there was no opening parenthesis between the two indices
                            break
                    next_index += 1
                indices_to_exclude.extend(range(index, next_index))  # add the indexes to delete their values
            index = next_index
        
        processed_tokens = [token for index, token in enumerate(tokens) if index not in indices_to_exclude]
        return processed_tokens


    def __process_boolean_tokens_delete_colon_and_keys(self, tokens: list[str]) -> list[str]:
        prev_index = 0
        indices_to_remove = []
        for index, token in enumerate(tokens):
            if token == ':':
                prev_index = index
                while prev_index > 0:
                    if tokens[prev_index - 1] in ['AND', 'OR', '(']:
                        break
                    else:
                        prev_index -= 1
                indices_to_remove.extend(range(prev_index, index+1))  # add the indexes to delete their values
        processed_tokens = [token for index, token in enumerate(tokens) if index not in indices_to_remove]
        return processed_tokens


    def __join_term_tokens_by_underscores(self, tokens: list[str]) -> list[str]:
        if len(tokens) <= 1:
            return tokens
        else:
            _2d_matrix_to_join_by_underscore = []
            for index, token in enumerate(tokens):
                if (token in ['AND', 'OR', ')']) or (index == (len(tokens)-1)): #if token is operator or its the last element
                    prev_index = index - 1
                    while prev_index >= 0:
                        if tokens[prev_index] in ['AND', 'OR', '(', ')']:
                            prev_index += 1
                            break
                        if prev_index == 0:
                            break
                        prev_index -= 1
                    last_index = index - 1 if token in ['AND', 'OR', ')'] else index
                    tuple_to_join_by_underscore = (prev_index, last_index)
                    if (tuple_to_join_by_underscore[1] - tuple_to_join_by_underscore[0]) > 0:
                        _2d_matrix_to_join_by_underscore.append(tuple_to_join_by_underscore)
            
            current_index = 0
            processed_tokens = []
            for start, end in _2d_matrix_to_join_by_underscore:
                # Make sure indexes are valid
                if start < 0 or end >= len(tokens) or start > end:
                    raise ValueError(f"Invalid indexes for joining elements: {start}, {end}")

                # Add the token elements before the current subset
                while current_index < start:
                    processed_tokens.append(tokens[current_index])
                    current_index += 1

                # Join subset elements with underscores
                joined_element = '_'.join(tokens[start:end+1])
                processed_tokens.append(joined_element)

                # Move the current index to the end of the subset
                current_index = end + 1
            
            # Agregar los elementos restantes de tokens después del último subconjunto
            while current_index < len(tokens):
                processed_tokens.append(tokens[current_index])
                current_index += 1       

        return processed_tokens