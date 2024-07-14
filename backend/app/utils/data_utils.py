from typing import Any
import os
import json



class DataUtils:

    @staticmethod
    def load_stopwords() -> tuple[str]:
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
        current_directory = os.path.dirname(__file__)
        data_file_path = os.path.join(current_directory, '..', 'data', 'test_data.json')

        # Validate if the path exists
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"File '{data_file_path}' does not exist.")

        with open(data_file_path) as f:
            test_data = json.load(f)
        return test_data