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





# Required Libraries

- GraphViz
- Python
    - numpy
    - scikit_learn
    - nltk
    - textblob
    - fastapi
    - uvicorn

