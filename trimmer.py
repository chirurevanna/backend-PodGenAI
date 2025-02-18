from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse
from pydub import AudioSegment
import os

app = FastAPI()

@app.post("/trim-audio")
async def trim_audio(file: UploadFile = File(...), start_time: str = Query(..., description="Start time in MM:SS format")):
    try:
        # Validate and parse the input time
        try:
            minutes, seconds = map(int, start_time.split(":"))
            if minutes < 0 or seconds < 0 or seconds >= 60:
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid time format. Use MM:SS.")

        # Convert start time to milliseconds
        start_time_in_ms = (minutes * 60 + seconds) * 1000
        end_time_in_ms = start_time_in_ms + (10 * 60 * 1000)  # Add 10 minutes to start time

        # Save the uploaded file to a temporary location
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Load the audio file using pydub
        audio = AudioSegment.from_file(temp_file_path)

        # Ensure the start and end times are within the audio duration
        if start_time_in_ms >= len(audio):
            raise HTTPException(status_code=400, detail="Start time exceeds audio duration.")
        if end_time_in_ms > len(audio):
            end_time_in_ms = len(audio)

        # Trim the audio between the calculated start and end time
        trimmed_audio = audio[start_time_in_ms:end_time_in_ms]

        # Save the trimmed audio to a temporary MP3 file
        trimmed_file_path = f"trimmed_{file.filename}"
        trimmed_audio.export(trimmed_file_path, format="mp3")

        # Delete the temporary input file
        os.remove(temp_file_path)

        # Return the trimmed audio file as a response
        return FileResponse(
            trimmed_file_path,
            media_type="audio/mpeg",
            filename=f"trimmed_{file.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
