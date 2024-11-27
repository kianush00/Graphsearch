from numpy.linalg import norm
from numpy import dot



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
    def get_cosine_between_vectors(
            vector1: list[float], 
            vector2: list[float]
            ) -> float:
        """
        Calculate and return the cosine of the angle between the two given vectors

        Parameters
        ----------
        vector1 : list[float]
            First vector to calculate
        vector2 : list[float]
            Second vector to calculate
        
        Returns
        -------
        cosine_of_angle : float
            The cosine of the angle between the two vectors
        """
        cosine_of_angle: float = 0.0
        
        # Avoid division by zero if the norm of any vector equals zero
        if norm(vector1) > 0 and norm(vector2) > 0:
            cosine_of_angle = dot(vector1, vector2) / norm(vector1) / norm(vector2)
        
        return cosine_of_angle

    
    
    @staticmethod
    def get_positions_sorted_desc(values: list[float]) -> list[int]:
        """
        Sort the given list of values in descending order and return a list of their indices.

        Parameters:
        ----------
        values : list[float]
            A list of float values to be sorted in descending order.

        Returns:
        -------
        list[int]
            A list of integers representing the sorted indices of the input list.
        """
        # Enumerate the original list to keep track of the original indices
        indexed_values = list(enumerate(values))
        
        # Sort the list of tuples (index, value) by the value in descending order
        sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)
        
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
    
    
    @staticmethod
    def calculate_distance_score_list(
            distance_occurrence_list: list[int]
            ) -> list[float]:
        """
        Calculate a distance score list from a list of distance occurrences, using the following
        formula:  1 / (4 ^ (distance - 1) )

        Parameters:
        distance_occurrence_list (list[int]): A list of distance occurrences (E.g. [1, 3, 3] list obtained
        from [1, 0, 2, 0] vicinity matrix value)

        Returns:
        list[float]: A list of transformed distance values.
        """
        distance_score_calculation_list: list[float] = [1 / (2 ** (distance - 1)) for distance in distance_occurrence_list]
        return distance_score_calculation_list
    
    
    @staticmethod
    def split_and_extend_from_underscore_values(strings_array: list[str]) -> list[str]:
        """
        This function takes an array of strings as input and splits each string at underscores.
        It then extends the result array with the split parts, excluding any empty strings.
        If a string does not contain underscores, it is appended directly to the result array.
        
        Parameters:
        array (list[str]): A list of strings to be processed.
        
        Returns:
        list[str]: A new list of strings, where each string from the input array is either split at underscores
                   and extended into the result array, or appended directly if it does not contain underscores.
        """
        result = []
        for word in strings_array:
            # Check if the word contains underscores
            if '_' in word:
                # Split the word by underscores, filter out empty strings, and extend the result array
                result.extend([part for part in word.split('_') if part != ''])
            else:
                # Append the word directly if it has no underscores
                result.append(word)
        return result

