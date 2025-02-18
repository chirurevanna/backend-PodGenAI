from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
import jiwer
import uvicorn

app = FastAPI()

# Define the transformation pipeline
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])





@app.post("/calculate-wer")
async def calculate_wer(
    reference: str = Form(...),
    hypothesis: str = Form(...)
):
    try:
        # Calculate the WER using the custom transformation
        wer_response = jiwer.wer(
            reference,
            hypothesis,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )

        # Calculate detailed measures with the same transformations
        measures = jiwer.compute_measures(
            reference,
            hypothesis,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )

        # Calculate lengths
        reference_length = sum(len(sentence) for sentence in reference)
        hypothesis_length = sum(len(sentence) for sentence in hypothesis)
        return[
            round(measures.get('wer', 0), 3),
            round(measures.get('mer', 0), 3),
            round(measures.get('wil', 0), 3),
            round(measures.get('wip', 0), 3),
            round(measures.get('cer', 0), 3),
            reference_length,
            hypothesis_length,
            measures.get('insertions'),
            measures.get('deletions'),
            measures.get('substitutions')
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while calculating WER: {str(e)}"
        )


