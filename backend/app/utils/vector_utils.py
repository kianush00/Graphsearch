from numpy.linalg import norm
from numpy import dot



class VectorUtils:

    @staticmethod
    def normalize_vector(
            vector: list[float], 
            new_min: float = 0, 
            new_max: float = 1.0
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
        
        Returns
        -------
        normalized_vector : list[float]
            The normalized vector
        """
        # Get the previous vector range
        old_min = min(vector)
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
        # Avoid division by zero if the norm of any vector equals zero
        if norm(vector1) > 0 and norm(vector2) > 0:
            cosine_of_angle = dot(vector1, vector2) / norm(vector1) / norm(vector2)
        else:
            cosine_of_angle = 0
        
        return cosine_of_angle