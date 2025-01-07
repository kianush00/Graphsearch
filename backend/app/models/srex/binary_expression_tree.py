from collections import deque 
import re
from copy import deepcopy
from functools import reduce
from utils.text_utils import TextUtils
from models.srex.vicinity_graph import VicinityGraph




class BinaryTreeNode:
    def __init__(self, value: str):
        """
        Initialize a new instance of BinaryTreeNode.
        
        Parameters:
        value (str): The value to be assigned to the node.
        """
        self.value = value
        self.graph: VicinityGraph | None = None
        self.left: BinaryTreeNode | None = None
        self.right: BinaryTreeNode | None = None
    

    def is_leaf(self) -> bool:
        """
        Check if the current node is a leaf node.
        
        Returns:
        bool: True if the node is a leaf node, False otherwise.
        """
        is_leaf = not self.left and not self.right
        return is_leaf
    

    def get_values_from_leaves(self) -> list[str]:
        """
        Retrieve the values of all leaf nodes in the subtree rooted at the current node.
        
        Returns:
        list[str]: A list of values from the leaf nodes.
        """
        if self.is_leaf():
            return [self.value]
        return self.left.get_values_from_leaves() + self.right.get_values_from_leaves()
    

    def get_leaves(self) -> list['BinaryTreeNode']:
        """
        Retrieve all leaf nodes in the subtree rooted at the current node.
        
        Returns:
        list['BinaryTreeNode']: A list of leaf nodes.
        """
        if self.is_leaf():
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()
    

    def get_graph_from_subtree_by_subquery(self, query: str) -> VicinityGraph | None:
        """
        Recursively search for a VicinityGraph object with a specific subquery in the 
        subtree rooted at the current node.
        
        Parameters:
        query (str): The subquery to search for.

        Returns:
        graph (VicinityGraph | None): The VicinityGraph object with the specified subquery, or None if no such object is found.
        """
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
        subtrees, calculating the sum of proximity ponderations and frequency ponderations, 
        that is, the union between each graph (tree node) of both subtrees.
        
        Parameters:
        external_subtree (BinaryTreeNode): The external subtree to be united

        Returns:
        copy_subtree (BinaryTreeNode): The union between copy of the subtree itself 
        and an external subtree.
        """
        copy_subtree = deepcopy(self)
        copy_subtree.__do_union_between_copy_self_and_subtree(external_subtree)
        return copy_subtree
    

    def do_graph_operation_from_subtrees(self) -> None:
        """
        Perform a graph operation on the subtrees of the current node.
        This function is intended to be called on non-leaf nodes of a tree. It validates that the tree does not contain
        any empty vicinity graphs in its leaf nodes. If no exceptions are raised, it sets and retrieves the graph obtained from
        the operation on the subtrees.

        Raises:
        Exception: If an error occurs during the graph operation on the subtrees.
        """
        if not self.is_leaf():
            #Validate if the tree has no empty vicinity graphs in leaves
            try:
                self.__set_and_get_graph_operated_from_subtrees()
            except Exception as e:
                print('Error at do_graph_operation_from_subtrees(): ' + repr(e))
    

    def __str__(self) -> str:
        """
        Return a string representation of the binary expression tree.
        
        The string representation is obtained by performing an in-order traversal of the tree.
        In-order traversal visits the left subtree, then the root, and finally the right subtree.
        
        Returns:
        str: A string representation of the binary expression tree.
        """
        return ''.join(self.__inorder_traversal())
    

    def tree_str(self) -> str:
        """
        Generate a string representation of the binary expression tree.

        This function uses the private method `__generate_tree_str()` to construct a string representation of the tree.
        The string representation is formatted with indentation to represent the tree structure. Each node in the tree
        is represented on a new line, with indentation based on its level in the tree.

        Returns:
        str: A string representation of the binary expression tree.
        """
        return self.__generate_tree_str()
    

    def __do_union_between_copy_self_and_subtree(self, 
            external_peer_node: 'BinaryTreeNode'
            ) -> None:
        """
        Unites an external subtree with the own copy subtree and modifies 
        the own subtree graphs. This method should only be used by the 
        copy from the original BinaryTreeNode object.
        
        Parameters:
        external_peer_node (BinaryTreeNode): The external peer node to be united
        """
        #Checks if both nodes have a non-null graph attribute, to then join them and set 
        #them in the copy attribute
        if self.graph and external_peer_node.graph:
            sum_scores = True   # Sum the proximity scores of both graphs, for intersected graph nodes
            current_node_united_graph = self.graph.get_union_to_graph(external_peer_node.graph, sum_scores)
            self.graph = current_node_united_graph
        
        #Checks if both nodes have a left node, to continue the recursion
        if self.left and external_peer_node.left:
            self.left.__do_union_between_copy_self_and_subtree(external_peer_node.left)
        
        #Checks if both nodes have a right node, to continue the recursion
        if self.right and external_peer_node.right:
            self.right.__do_union_between_copy_self_and_subtree(external_peer_node.right)


    def __set_and_get_graph_operated_from_subtrees(self) -> VicinityGraph:
        """
        This function sets and gets the graph operated from the left and right subtrees.
        It performs an intersection or union operation based on the boolean operator value.

        Returns:
        VicinityGraph: The graph resulting from the intersection or union operation.
        """
        left_graph = self.__get_leaf_graph(self.left)
        right_graph = self.__get_leaf_graph(self.right)

        # If the left and right subtrees have the same subquery, the function returns the left graph.
        if left_graph.subquery == right_graph.subquery:
            graph = left_graph
        # If the boolean operator is 'AND' or 'OR', it returns the union of the left and right graphs.
        elif self.value in ['AND', 'OR']:
            # The resulting proximity ponderation value is the max value between the left and right graphs.
            sum_scores = False  # Get the maximum proximity score between both graphs, for intersected graph nodes
            graph = left_graph.get_union_to_graph(right_graph, sum_scores)
        # If the boolean operator is neither 'AND' nor 'OR', it raises a ValueError.
        else:
            raise ValueError(f"Invalid operator value: {self.value}")

        #The new graph inherits the same subquery from the self graph
        if self.graph:
            graph.subquery = self.graph.subquery

        self.graph = graph
        return graph
    
    
    def __get_leaf_graph(self, node: 'BinaryTreeNode') -> VicinityGraph:
        """
        Retrieve the vicinity graph associated with a leaf node in the binary expression tree.

        Parameters:
        ----------
        node : BinaryTreeNode
            The node for which the vicinity graph needs to be retrieved.

        Returns:
        -------
        VicinityGraph
            The vicinity graph associated with the leaf node. If the node is not a leaf node,
            a TypeError is raised. If the vicinity graph is not set for the leaf node,
            a TypeError is also raised.

        Raises:
        -------
        TypeError
            If the node is not a leaf node or if the vicinity graph is not set for the leaf node.
        """
        if node.is_leaf():
            if node.graph:
                return node.graph
            else:
                raise TypeError('Null vicinity graphs in leaves')
        else:
            return node.__set_and_get_graph_operated_from_subtrees()
    

    def __inorder_traversal(self, level: int = 0) -> list[str]:
        """
        Perform an in-order traversal of the binary tree and return a list of strings representing the nodes.
        The traversal includes opening and closing parentheses for each level of the tree.
        The function appends spaces around boolean operators ('AND' and 'OR') and does not add spaces around terms.

        Parameters:
        level (int): The current level of the tree. Default is 0 (root level).

        Returns:
        result (list[str]): A list of strings representing the nodes in in-order traversal.
        """
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
        """
        Generate a string representation of the binary tree.

        Parameters:
        level (int): The current level of recursion. It is used to indent the output string.

        Returns:
        str: A string representation of the binary tree. Each node is represented on a new line,
            with indentation based on its level in the tree.
        """
        output = "\t" * level + self.value + "\n"
        if self.left:
            output += self.left.__generate_tree_str(level + 1)
        if self.right:
            output += self.right.__generate_tree_str(level + 1)
        return output



class BinaryExpressionTree:
    def __init__(self, raw_query: str):
        """
        Initialize a BinaryExpressionTree object with a given boolean query.

        Parameters:
        raw_query (str): The boolean query to be parsed and used to construct the tree.
        """
        self.__raw_query = raw_query
        self.root = self.__get_tree_built(self.__raw_query)
    

    @property
    def raw_query(self) -> str:
        """
        Retrieve the raw boolean query used to construct the BinaryExpressionTree.

        Returns:
        str: The raw boolean query used to construct the BinaryExpressionTree.
        """
        return self.__raw_query
    

    def get_query_terms_str_list_with_underscores(self) -> list[str]:
        """
        Get a list of query terms from the binary expression tree, with underscores.
        This function retrieves the values from the leaves of the binary expression tree,
        which represent the query terms. The values are returned as a list of strings,
        with underscores included.

        Returns:
        list[str]: A list of query terms from the binary expression tree, with underscores.
        """
        return self.root.get_values_from_leaves()
    
    
    def get_individual_query_terms_str_list(self) -> list[str]:
        """
        Get a list of individual query terms from the binary expression tree, without underscores.
        This function retrieves the values from the leaves of the binary expression tree,
        which represent the query terms. The values are returned as a list of strings.

        Returns:
        list[str]: A list of query terms from the binary expression tree, without underscores.
        """
        query_terms_with_underscores = self.root.get_values_from_leaves()
        
        individual_query_terms = []
        for string in query_terms_with_underscores:
            if "_" in string:
                individual_query_terms.extend(string.split("_"))  # Dividimos y añadimos los sub-elementos a la lista final
            else:
                individual_query_terms.append(string)  # Si no tiene underscores, lo añadimos tal cual
        return individual_query_terms
    

    def get_query_terms_as_leaves(self) -> list[BinaryTreeNode]:
        """
        Retrieve the query terms from the binary expression tree as leaf nodes.

        Returns:
        -------
        list[BinaryTreeNode]:
            A list of BinaryTreeNode objects representing the query terms.
            Each BinaryTreeNode object corresponds to a leaf node in the binary expression tree.
        """
        return self.root.get_leaves()
    

    def get_graph(self) -> VicinityGraph | None:
        """
        Retrieve the graph associated with the root node of the binary expression tree.

        Returns:
        VicinityGraph | None:
            The graph associated with the root node of the binary expression tree.
            If no graph is associated with the root node, the function returns None.
        """
        return self.root.graph
    

    def get_graph_by_subquery(self, query: str) -> VicinityGraph | None:
        """
        Get the graph associated with a specific subquery from the root node of the tree.

        Parameters:
        ----------
        query : str
            The subquery to search for in the tree.

        Returns:
        -------
        VicinityGraph | None
            The graph associated with the subquery if found, otherwise None.
            If the graph is not found, a message is printed to the console.
        """
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
        copy_tree = deepcopy(self)
        
        #Get the deep union between the copy root node and the external root node
        copy_root = copy_tree.root.get_union_to_subtree(external_tree.root)
        copy_tree.root = copy_root
        return copy_tree
    

    def operate_non_leaf_graphs_from_leaves(self) -> None:
        """
        Operate the graphs associated with each non-leaf node in the binary expression tree.

        This function iterates through the non-leaf nodes of the binary expression tree,
        retrieves the associated graph for each leaf node, and performs a specific operation
        on the graph (in this case, it is assumed that the operation is defined in the
        `do_graph_operation_from_subtrees` method of the `BinaryTreeNode` class).

        Returns:
        -------
        None
            The function does not return any value. It modifies the graphs associated with 
            the non-leaf nodes in-place.
        """
        self.root.do_graph_operation_from_subtrees()
    

    def do_text_transformations_to_query_terms(self, 
            stop_words: tuple[str] = (), 
            lema: bool = True, 
            stem: bool = False
            ) -> None:
        """
        Apply text transformations to the terms in the leaf nodes.

        Parameters
        ----------
        stop_words : tuple[str], optional
            List of stop words to be removed from the terms
        lema : bool, optional
            If True, lemmatization is applied
        stem : bool, optional
            If True, stemming is applied
        """
        def transform_node_if_leaf(node: BinaryTreeNode) -> None:
            if node.is_leaf():
                node.value = TextUtils.get_transformed_text_if_it_has_underscores(node.value, stop_words, 
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

        This function iterates through the binary expression tree, starting from the root node.
        For each node, it sets the graph attribute to None, effectively removing the graph associated with that node.
        The function uses recursion to handle nested nodes in the tree.

        Returns:
        None: The function does not return any value. It modifies the tree in-place.
        """
        def remove_graph(node: BinaryTreeNode):
            node.graph = None
            if node.left:
                remove_graph(node.left)
            if node.right:
                remove_graph(node.right)
        
        remove_graph(self.root)


    def __str__(self) -> str:
        """
        Return a string representation of the binary expression tree.
        The string representation is obtained by converting the root node of the tree to a string.

        Returns:
        str: A string representation of the binary expression tree.
        """
        return str(self.root)
    

    def tree_str(self) -> str:
        """
        Return a string representation of the binary expression tree.
        The string representation is obtained by converting the root node of the tree to a string.
        This function is used to visualize the tree structure and can be helpful for debugging purposes.

        Returns:
        str: A string representation of the binary expression tree.
        """
        return self.root.tree_str()
    
    
    def __get_tree_built(self, raw_query: str) -> BinaryTreeNode:
        """
        Build a binary expression tree from a boolean query.
        The function separates the boolean query into tokens, processes the tokens to obtain an infix token list,
        converts the infix token list to a postfix token list, and constructs a binary expression tree from the
        postfix token list. The constructed tree's root node is then returned.
        
        Parameters
        ----------
        raw_query: str
            The boolean query to be parsed.

        Returns
        -------
        root_node: BinaryTreeNode
            The root node of the constructed binary expression tree.
        """
        tokens = self.__separate_boolean_query(raw_query)
        infix_tokens = self.__process_infix_tokens(tokens)
        postfix_tokens = self.__infix_to_postfix(infix_tokens)
        root_node = self.__construct_tree_from_postfix(postfix_tokens)
        return root_node
    

    def __separate_boolean_query(self, query: str) -> list[str]:
        """
        Separate a boolean query into individual tokens.
        
        Parameters
        ----------
        query: str
            The boolean query to be separated.

        Returns
        -------
        tokens: list[str]
            A list of tokens extracted from the boolean query. Tokens include boolean 
            operators, parentheses, colons, and terms.
        """
        # Define a regex pattern to match boolean operators, parentheses, colons and terms
        pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\(|\)|\w+|:)'
        
        # Find all matches using the pattern
        tokens = re.findall(pattern, query)
        
        return tokens
    

    def __process_infix_tokens(self, tokens: list[str]) -> list[str]:
        """
        Process a list of boolean query tokens into an infix token list.
        
        Parameters
        ----------
        tokens: list[str]
            A list of boolean query tokens.

        Returns
        -------
        infix_tokens: list[str]
            A list of boolean query tokens in infix notation.
        
        Raises
        -------
        ValueError
            If the boolean expression is not valid.
        """
        validation_tuple = self.__validate_boolean_expression(tokens)
        if validation_tuple[0]:
            tokens_deleted_not_expressions = self.__process_boolean_tokens_delete_not_expressions(tokens)
            tokens_deleted_colon_and_keys = self.__process_boolean_tokens_delete_colon_and_keys(tokens_deleted_not_expressions)
            infix_tokens = self.__join_term_tokens_by_underscores(tokens_deleted_colon_and_keys)
            return infix_tokens
        else:
            raise ValueError(validation_tuple[1])
    

    def __infix_to_postfix(self, infix_tokens: list[str]) -> list[str]:
        """
        Convert a list of boolean query tokens in infix notation to postfix notation.
        
        Parameters
        ----------
        infix_tokens: list[str]
            A list of boolean query tokens in infix notation.

        Returns
        -------
        postfix_list: list[str]
            A list of boolean query tokens in postfix notation.
        """
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
        """
        Construct a binary expression tree from a postfix token list.
        
        Parameters
        ----------
        postfix_tokens: list[str]
            A list of boolean query tokens in postfix notation.

        Returns
        -------
        root_node: BinaryTreeNode
            The root node of the constructed binary expression tree.
        """
        stack: list[BinaryTreeNode] = []
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
        root_node = stack.pop()
        return root_node


    def __validate_boolean_expression(self, tokens: list[str]) -> tuple[bool, str]:
        """
        Validate a boolean expression from tokens, checking parenthesis syntax and operators syntax.
        
        Parameters
        ----------
        tokens: list[str]
            A list of boolean query tokens.

        Returns
        -------
        response: tuple[bool, str]
            A tuple with a boolean which indicates if the expression is valid, and a string with
            the details of the validation.
        """
        # Check parenthesis syntax
        valid, message = self.__validate_parenthesis(tokens)
        if not valid:
            return valid, message

        # Check AND OR operators
        valid, message = self.__validate_operators(tokens)
        if not valid:
            return valid, message

        return True, "The expression is valid"
    
    
    def __validate_parenthesis(self, tokens: list[str]) -> tuple[bool, str]:
        """
        Validate boolean query tokens to check parenthesis syntax.
        
        This function iterates through the tokens list, identifies opening and closing parentheses, and checks their syntax.
        It uses a stack to handle nested parentheses and correctly identifies the appropriate tokens.
        
        Parameters:
        tokens (list[str]): A list of boolean query tokens. Each token represents a word or a boolean operator.
        
        Returns:
        tuple[bool, str]: A tuple with a boolean indicating if the parenthesis syntax is valid, and a string with the details of the validation.
        """
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
        return True, ""
    
    
    def __validate_operators(self, tokens: list[str]) -> tuple[bool, str]:
        """
        Validate boolean query tokens to check operators syntax.
        
        This function iterates through the tokens list, identifies operators, and checks their syntax.
        It checks if operators are not at the start or end of the expression, and if they are surrounded by valid terms.
        
        Parameters:
        tokens (list[str]): A list of boolean query tokens. Each token represents a word or a boolean operator.
        
        Returns:
        tuple[bool, str]: A tuple with a boolean indicating if the operators syntax is valid, and a string with the details of the validation.
        """
        operators = {'AND', 'OR'}
        length = len(tokens)

        for i, token in enumerate(tokens):
            if token in operators:
                if i == 0 or i == length - 1:
                    return False, f"Operator {token} cannot be at the start or end of the expression"
                if tokens[i-1] in operators or tokens[i+1] in operators:
                    return False, f"Operator {token} must be surrounded by valid terms"
        return True, ""


    def __process_boolean_tokens_delete_not_expressions(self, tokens: list[str]) -> list[str]:
        """
        Process boolean query tokens to delete NOT expressions.
        
        This function iterates through the tokens list, identifies NOT expressions, and removes them.
        It uses a stack to handle nested NOT expressions and correctly removes the appropriate tokens.
        
        Parameters:
        tokens (list[str]): A list of boolean query tokens.
        
        Returns:
        list[str]: A list of boolean query tokens with NOT expressions removed.
        """
        indices_to_exclude = []
        index = 0

        while index < len(tokens):
            if tokens[index] == 'NOT':
                next_index = self.__process_not_expression(tokens, index)
                indices_to_exclude.extend(range(index, next_index)) # add the indexes to delete their values
            else:
                next_index = index + 1
            index = next_index

        processed_tokens = [token for index, token in enumerate(tokens) if index not in indices_to_exclude]
        return processed_tokens
    
    
    def __process_not_expression(self, tokens: list[str], index: int) -> int:
        """
        Process boolean query tokens to handle NOT expressions.
        
        This function iterates through the tokens list starting from the given index, identifies NOT expressions,
        and returns the index of the token following the NOT expression. It uses a stack to handle nested NOT expressions
        and correctly identifies the appropriate tokens.
        
        Parameters:
        tokens (list[str]): A list of boolean query tokens.
        index (int): The starting index from which to process the tokens.
        
        Returns:
        int: The index of the token following the NOT expression.
        """
        next_index = index + 1
        stack = 0
        while next_index < len(tokens):
            if tokens[next_index] == '(':
                stack += 1
            elif tokens[next_index] == ')':
                if stack < 2: # if it is the last closing parenthesis, or there's no opening parenthesis between the 2 indices
                    if stack == 1:
                        next_index += 1     # removes the closing parenthesis if there was an opening parenthesis before
                    return next_index
                stack -= 1
            elif (tokens[next_index] in ['AND', 'OR', 'NOT']) and (stack == 0):
                return next_index  # If there is an operator after NOT with the stack empty, then return the current next index
            next_index += 1
        return next_index


    def __process_boolean_tokens_delete_colon_and_keys(self, tokens: list[str]) -> list[str]:
        """
        Process boolean query tokens to delete colon (:) and key expressions.
        
        This function iterates through the tokens list, identifies colon (:) expressions, and removes them.
        It also identifies key expressions (tokens that follow a colon) and removes them.
        
        Parameters:
        tokens (list[str]): A list of boolean query tokens.
        
        Returns:
        list[str]: A list of boolean query tokens with colon (:) and key expressions removed.
        """
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
        """
        Join multi-word query terms in a boolean expression with underscores.
        
        This function takes a list of boolean query tokens and identifies multi-word query terms.
        It then joins these terms with underscores and returns a new list of tokens with the joined terms.
        
        Parameters:
        tokens (list[str]): A list of boolean query tokens. Each token represents a word or a boolean operator.
        
        Returns:
        list[str]: A new list of boolean query tokens with multi-word query terms joined by underscores.
        """
        if len(tokens) <= 1:
            return tokens
        
        tuples_to_join_by_underscore = self.__get_tuples_to_join_by_underscore(tokens)
        processed_tokens = self.__get_tokens_with_terms_joined_by_underscore(tokens, tuples_to_join_by_underscore)
        return processed_tokens
    
    
    def __get_tuples_to_join_by_underscore(self, tokens: list[str]) -> list[tuple[int, int]]:
        """
        This function identifies multi-word query terms in a boolean expression and returns a list of tuples,
        where each tuple represents the start and end index of a multi-word term in the tokens list.

        Parameters:
        tokens (list[str]): A list of boolean query tokens. Each token represents a word or a boolean operator.

        Returns:
        list[tuple[int, int]]: A list of tuples, where each tuple contains the start and end index of a multi-word term.
        """
        tuples_to_join_by_underscore: list[tuple[int, int]] = []
        start_index: int | None = None

        # Append tuples of index ranges for query terms with more than one word
        for index, token in enumerate(tokens):
            if token not in ['AND', 'OR', '(', ')']:  # if token is a term
                if start_index is None:   # if term is the first one
                    start_index = index
            elif start_index is not None:   # if token is an operand or parenthesis after a term
                if start_index < index - 1:  # the tuple must have a difference greater than zero
                    tuples_to_join_by_underscore.append((start_index, index - 1))
                start_index = None

        # Try to append last token elements if they are a multi word term
        if start_index is not None and (start_index < len(tokens) - 1):
            tuples_to_join_by_underscore.append((start_index, len(tokens) - 1))

        return tuples_to_join_by_underscore
    
    
    def __get_tokens_with_terms_joined_by_underscore(self, 
        tokens: list[str], 
        tuples_to_join: list[tuple[int, int]]
        ) -> list[str]:
        """
        This function takes a list of boolean query tokens and a list of tuples representing the start and end index of 
        multi-word terms. It then joins the multi-word terms with underscores and returns a new list of tokens with the joined 
        terms.

        Parameters:
        tokens (list[str]): A list of boolean query tokens. Each token represents a word or a boolean operator.
        tuples_to_join (list[tuple[int, int]]): A list of tuples, where each tuple contains the start and end index of a 
        multi-word term.

        Returns:
        list[str]: A new list of boolean query tokens with multi-word query terms joined by underscores.
        """
        current_index = 0
        processed_tokens = []
        for start, end in tuples_to_join:
            # Add the token elements before the current subset
            while current_index < start:
                processed_tokens.append(tokens[current_index])
                current_index += 1

            # Join subset elements with underscores
            joined_element = '_'.join(tokens[start:end+1])
            processed_tokens.append(joined_element)

            # Move the current index to the end of the subset
            current_index = end + 1
        
        # Add the remaining token elements after the last subset
        while current_index < len(tokens):
            processed_tokens.append(tokens[current_index])
            current_index += 1       

        return processed_tokens