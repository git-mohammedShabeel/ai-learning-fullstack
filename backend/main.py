from fastapi import FastAPI

app = FastAPI(title="AI Learning Assistant")

@app.get("/")
def root():
    return {"status": "ok", "project": "AI Learning Assistant - Full Stack"}