# SREX Frontend

This repository contains the frontend of the SREX project.


# Install dependencies

To install frontend dependencies, use the following command:

```bash
    npm install
```


# Build the project

To use TypeScript, the files located in the ./src directory must be compiled with output in ./dist. To do this, run the following command:

```bash
    npm run build:dev
```


# Run HTTP Server

1. For running the HTTP Server, execute the following command:

```bash
    npm start
```

2. Then go to [http://localhost:3000](http://localhost:3000)



# Run Static Analysis

Run the following command from the frontend/ directory to launch analysis:

### Windows

```bash
    sonar-scanner.bat -D"sonar.token=myAuthenticationToken"
```

### Linux / MacOS

```bash
    sonar-scanner -Dsonar.token=myAuthenticationToken
```

