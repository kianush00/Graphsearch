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
            factor = float((results_size - 0.7 * (ranking_position - 1)) / results_size)
        elif (weighted == 'inverse'):
            factor = float(1 / (1 + 0.05 * (ranking_position - 1)))
        else:
            factor = 1.0

        return factor
    
    
    @staticmethod
    def calculate_term_ponderation(
            term_frequency: int,
            document_weight: float
            ) -> float:
        """
        Calculate the term ponderation based on the term frequency and document weight.

        Parameters:
        term_frequency (int): The frequency of the term in the document.
        document_weight (float): The weight factor of the document, calculated using the MathUtils.calculate_document_weight method.

        Returns:
        float: The term ponderation, calculated as (1 + log10(term_frequency)) * document_weight.
        """
        ponderation = (1 + math.log10(term_frequency)) * document_weight
        return ponderation