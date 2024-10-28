import math


class MathUtils:

    @staticmethod
    def calculate_document_weight(
            results_size: int, 
            ranking_position: int,
            weighted: str = 'linear'
            ) -> float:
        """
        Calculate a weight factor depending on the argument value ('linear' or 'inverse'), and 
        the position of the document in the ranking.

        Parameters
        ----------
        results_size : int
            The total number of results/documents.
        ranking_position : int
            The position of the document in the ranking.
        weighted : str
            The type of weighting to be applied. Can be 'linear', 'inverse', or any other value.

        Returns
        -------
        factor : float
            The calculated weight factor.
        """
        if (weighted == 'linear'):
            factor = float((results_size - 0.5 * (ranking_position - 1)) / results_size)
        elif (weighted == 'inverse'):
            factor = float(1 / (1 + 0.05 * (ranking_position - 1)))
        else:
            factor = 1.0

        return factor
    
    
    @staticmethod
    def calculate_term_proximity_score(
            score_list_sum: float,
            document_weight: float
            ) -> float:
        """
        Calculate the term proximity score based on the score list sum and document weight.

        Parameters:
        score_list_sum (float): The sum of the list of distance scores calculated previously.
        document_weight (float): The weight factor of the document, calculated previously.

        Returns:
        float: The term proximity score, calculated as (1 + log10(score_list_sum)) * document_weight.
        """
        # Validate the score list sum parameter
        if score_list_sum < 1.0:
            return -1.0
        
        prox_score = (1 + math.log10(score_list_sum)) * document_weight
        return prox_score