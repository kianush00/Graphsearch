class MissingEnvironmentVariableError(Exception):
    """
    Raised when a required environment variable is missing.
    """
    def __init__(self, message: str):
        super().__init__(message)