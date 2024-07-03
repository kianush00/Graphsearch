# SREX Library

This reposotory include the libraries, tests and notebooks of the **S**earch **R**esults **EX**plorer application.

- **srex.py :** basic library including basic methods for text processing an graphs construction.

- **srex_classes.py:** advanced library which includes high level functionality

- **test_term_distance_functions-v\*.ipynb :** Jupyter notebooks implementing the srex functionality

- **Unittest\*.ipynb :** Jupyter notebooks applying some unit tests to the methods defined in the basic library (srex.py)


# Run Uvicorn API

For running Uvicorn API, execute the following command:

```bash
    uvicorn app.main:app --host 127.0.0.1 --port 8080
```

# Run Tests

Tests are run through the VSCode editor. For the above, follow the steps below:

1. On the left side panel, select the `Testing` option

2. Run Tests



# Required Libraries

- GraphViz
- Python
    - numpy
    - scikit_learn
    - nltk
    - textblob
    - fastapi
    - uvicorn

