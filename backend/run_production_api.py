import os
import sys
from gunicorn.app.base import BaseApplication
from uvicorn.workers import UvicornWorker


class GunicornApplication(BaseApplication):
    """Custom Gunicorn application to run FastAPI with Uvicorn workers."""

    def __init__(self, app_module, options=None):
        self.options = options or {}
        self.application = app_module
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if value is not None}
        for key, value in config.items():
            self.cfg.set(key, value)

    def load(self):
        return self.application


# Add the 'app' subdir to the sys.path
app_path = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_path)

if __name__ == "__main__":
    from app.main import app  # Import the FastAPI app

    options = {
        "bind": "0.0.0.0:8080",
        "workers": 4,
        "worker_class": "uvicorn.workers.UvicornWorker",
    }

    GunicornApplication(app, options).run()
