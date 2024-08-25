import math



class VectorUtils:

    @staticmethod
    def normalize_vector(
            vector: list[float], 
            new_min: float = 0, 
            new_max: float = 1.0,
            old_min: float | None = None,
            old_max: float | None = None
            ) -> list[float]:
        """
        Normalize the given vector to the new range [new_min, new_max] and return the normalized vector.

        Parameters
        ----------
        vector : list[float]
            Vector to normalize
        new_min : float, optional
            Minimum value of the new range to normalize
        new_max : float, optional
            Maximum value of the new range to normalize
        old_min : float, optional
            Minimum value of the original range to normalize
        old_max : float, optional
            Maximum value of the original range to normalize
        
        Returns
        -------
        normalized_vector : list[float]
            The normalized vector
        """
        # Validate the vector
        if (not vector) or (new_max < new_min):
            return []
        
        # If the user didn't specify the old values, then get the previous vector range
        if old_min is None:
            old_min = min(vector)
        if old_max is None:
            old_max = max(vector)
        
        # Avoid division by zero if the vector has all the same values
        if old_max == old_min:
            return [new_min for _ in vector]
        
        # Normalize the vector to the new range using the formula:
        normalized_vector = [
            new_min + (new_max - new_min) * (value - old_min) / (old_max - old_min)
            for value in vector
        ]
        
        return normalized_vector
    
    
    @staticmethod
    def get_euclidean_distance(vector1: list[float], vector2: list[float]) -> float:
        """
        Calculates the Euclidean distance between two vectors.
        
        Parameters
        ----------
        vector1 : list[float]
            First vector.
        vector2 : list[float] 
            Second vector.

        Returns
        -------
        distance : float
            Euclidean distance between the two vectors.
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have the same length.")

        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))
        return distance
    
    
    @staticmethod
    def get_positions_sorted_asc(values: list[float]) -> list[int]:
        """
        Sort the given list of values in ascending order and return a list of their indices.

        Parameters:
        ----------
        values : list[float]
            A list of float values to be sorted.

        Returns:
        -------
        list[int]
            A list of integers representing the sorted indices of the input list.
        """
        # Enumerate the original list to keep track of the original indices
        indexed_values = list(enumerate(values))
        
        # Sort the list of tuples (index, value) by the value in ascending order
        sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1], reverse=False)
        
        # Extract the sorted indices
        sorted_positions = [index for index, value in sorted_indexed_values]
        
        return sorted_positions
    
    
    @staticmethod
    def calculate_distances_between_term_positions(
            term1_positions: list[int], 
            term2_positions: list[int],
            limit_distance: int
            ) -> list[int]:
        """
        Compare the positions vectors of two terms, and return the list of 
        frequencies per distance between query terms and vicinity terms
        
        E.g.\n
        term1_positions = [0, 2, 4, 6]\n
        term2_positions = [1, 3, 5, 7]\n
        limit_distance = 7\n
        result = [7, 0, 5, 0, 3, 0, 1]\n

        Parameters
        ----------
        term1_positions : list[int]
            List of positions of the first term
        term2_positions : list[int]
            List of positions of the second term
        limit_distance : int
            Limit distance to calculate between the two term positions lists

        Returns
        -------
        frequencies_per_distance : list[int]
            List of frequencies per distance between query terms and vicinity terms
        """
        frequencies_per_distance = [0] * limit_distance

        for term1_pos in term1_positions:
            for term2_pos in term2_positions:
                absolute_distance = abs(term1_pos-term2_pos)
                if (absolute_distance <= limit_distance):
                    frequencies_per_distance[absolute_distance-1] += 1

        return frequencies_per_distance
