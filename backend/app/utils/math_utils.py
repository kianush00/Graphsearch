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
            if results_size <= 1:
                factor = 1.0
            else:        
                factor = float(0.5 + 0.5 * (1 - ((ranking_position - 1) / (results_size - 1))))
        elif (weighted == 'inverse'):
            factor = float(1 / (1 + 0.05 * (ranking_position - 1)))
        else:
            factor = 1.0

        return factor
