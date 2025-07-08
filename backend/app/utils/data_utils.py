from typing import Any, Tuple
import os
import json
import re


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
        
        DataUtils._validate_path_exists(data_file_path)

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

        DataUtils._validate_path_exists(data_file_path)

        with open(data_file_path) as f:
            test_data = json.load(f)
        return test_data
    
    @staticmethod
    def load_json_file(filename: str) -> Any:
        """
        Loads a JSON file from the specified file name.

        Args:
            filename (str): The name of the JSON file to be loaded.

        Returns:
            Any: The data loaded from the JSON file.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
        """
        current_directory = os.path.dirname(__file__)
        data_file_path = os.path.join(current_directory, '..', 'data', filename)

        DataUtils._validate_path_exists(data_file_path)

        with open(data_file_path, encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data
    
    
    @staticmethod
    def extract_articles_from_json_results(results_data: dict[str, Any]) -> list[dict[str, str]]:
        """
        From the loaded results data, build and return a list of dicts
        containing only:
        - 'title' ← the original 'content' field
        - 'article_number' ← the original 'docid' field

        Args:
            results_data: Dict with a key "results" mapping to a list of result dicts.

        Returns:
            A new list of dictionaries, in the same order, each with keys:
                'title': str
                'article_number': str
        """
        extracted = []
        for result in results_data.get("results", []):
            raw = result.get("content", "")
            # 1) Split on newlines/tabs, drop empty
            tokens = re.split(r'[\n\t]+', raw)
            tokens = [t for t in tokens if t.strip()]

            # 2) In first 4 tokens, strip off any "prefix -- " or "prefix _ "
            for i in range(min(4, len(tokens))):
                for sep in (" -- ", " _ "):
                    if sep in tokens[i]:
                        # keep only what comes after the separator
                        tokens[i] = tokens[i].split(sep, 1)[1].strip()
                        break  # no need to test the other sep

            # 3) Rejoin and final clean
            cleaned = " ".join(tokens)
            cleaned = cleaned.replace('\n', ' ').replace('\t', '').replace('\"', '')

            extracted.append({
                "title": cleaned,
                "article_number": result.get("docid", "")
            })
        return extracted
    
    
    @staticmethod
    def write_dict_to_json(data: dict[str, Any], filename: str) -> None:
        """
        Serializes a Python dictionary to a JSON file.

        Args:
            data:       The dictionary to write.
            filename:   The filename where to save the JSON.
        """
        current_directory = os.path.dirname(__file__)
        output_path = os.path.join(current_directory, '..', 'data', filename)

        # Write out the JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    
    @staticmethod
    def _validate_path_exists(path: str) -> None:
        """
        Checks that the file exists at 'path'; raises FileNotFoundError otherwise.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' does not exist.")