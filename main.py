from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional

app = FastAPI()

@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...)
):
    try:
        # Read the file content
        content = await file.read()
        text = content.decode('utf-8')

       
        summary = "test summary" 

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))