import os
import uvicorn
import sys

# Add the 'app' subdir to the sys.path
app_path = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_path)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host='0.0.0.0', port=8080)