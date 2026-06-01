from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import predict_animals

app = FastAPI(
    title="Pashu Drishti API",
    description="High-performance asynchronous backend API for real-time animal object detection.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_animals(file: UploadFile = File(...)):
    contents = await file.read()
    results = predict_animals(contents)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)