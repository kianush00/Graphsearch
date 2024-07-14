from typing import Any, Tuple
import os
import json



class DataUtils:

    @staticmethod
    def load_stopwords() -> Tuple[str]:
        """
        Loads stopwords from a JSON file located in the data directory.
        
        Returns:
            Tuple[str]: A tuple containing the stopwords.
        
        Raises:
            FileNotFoundError: If the stopwords_data.json file does not exist.
        """
        current_directory = os.path.dirname(__file__)
        data_file_path = os.path.join(current_directory, '..', 'data', 'stopwords_data.json')
        
        # Validate if the path exists
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"File '{data_file_path}' does not exist.")

        with open(data_file_path) as f:
            stopwords_data = json.load(f)
        stop_words_list = tuple(stopwords_data.get('words'))
        return stop_words_list
    
    
    @staticmethod
    def load_test_data() -> Any:
        """
        Loads test data from a JSON file located in the data directory.
        
        Returns:
            Any: The test data loaded from the file.
        
        Raises:
            FileNotFoundError: If the test_data.json file does not exist.
        """
        current_directory = os.path.dirname(__file__)
        data_file_path = os.path.join(current_directory, '..', 'data', 'test_data.json')

        # Validate if the path exists
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"File '{data_file_path}' does not exist.")

        with open(data_file_path) as f:
            test_data = json.load(f)
        return test_data