# SREX Library

This reposotory include the libraries, tests and notebooks of the **S**earch **R**esults **EX**plorer application.

- **srex.py :** basic library including basic methods for text processing an graphs construction.

- **srex_classes.py:** advanced library which includes high level functionality

- **test_term_distance_functions-v\*.ipynb :** Jupyter notebooks implementing the srex functionality

- **Unittest\*.ipynb :** Jupyter notebooks applying some unit tests to the methods defined in the basic library (srex.py)


# Run Uvicorn API

For running Uvicorn API, execute the following command:

```bash
    python run_api.py
```

# Run Tests

For running tests, execute the following command:

```bash
    python run_tests.py
```

# Run Static Analysis

Run the following command from the backend/ directory to launch analysis:

### Windows

```bash
    sonar-scanner.bat -D"sonar.token=myAuthenticationToken"
```

### Linux / MacOS

```bash
    sonar-scanner -Dsonar.token=myAuthenticationToken
```


# ENV File

Create a `.env` file in the current backend directory of the project with the following content:

```bash
    IEEE_XPLORE_API_KEY=your_api_key
```


# Required Libraries

- graphviz
- numpy
- scikit_learn
- nltk
- textblob
- fastapi
- uvicorn
- pydantic
- python-dotenv



# API Endpoints

| Endpoint                | Method | Description                | Classes/Services involved                          |
|-------------------------|--------|----------------------------|----------------------------------------------------|
| /get-ranking            | POST   | Get ranking from query     | Ranking, VicinityGraph, VicinityNode               |
| /get-ranking-example-1  | POST   | Get ranking 1st example    | Ranking, VicinityGraph, VicinityNode               |
| /get-ranking-example-2  | POST   | Get ranking 2nd example    | Ranking, VicinityGraph, VicinityNode               |
| /rerank                 | POST   | Rerank the current ranking | Ranking, VicinityGraph, VicinityNode, VectorUtils  |
