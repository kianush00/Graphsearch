import uvicorn
import sys

sys.path.insert(0, "./app")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host='127.0.0.1', port=8080)