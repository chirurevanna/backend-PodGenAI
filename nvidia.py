from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import time
import torch
from pydub import AudioSegment
from nemo.collections.asr.models import EncDecMultiTaskModel

app = FastAPI()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.empty_cache()  # Clear CUDA cache
torch.cuda.synchronize()  # Sync GPU

# Load NVIDIA Canary-1B Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
canary_model.to(device)

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    source_lang: str = "en",
    target_lang: str = "en",
    task: str = "asr"
):
    try:
        # Save the uploaded audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Convert MP3 to WAV (16 kHz mono)
        audio = AudioSegment.from_file(temp_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            audio.export(wav_file.name, format="wav")
            wav_path = wav_file.name

        start_time = time.time()

        # Run Transcription / Translation
        try:
            transcript = canary_model.transcribe(
                audio=[wav_path],
                batch_size=1,
                task=task,
                source_lang=source_lang,
                target_lang=target_lang,
                pnc="yes"
            )
        except RuntimeError as e:
            print(f"CUDA error: {e}. Switching to CPU.")
            canary_model.to("cpu")  # Move the model to CPU
            transcript = canary_model.transcribe(
                audio=[wav_path],
                batch_size=1,
                task=task,
                source_lang=source_lang,
                target_lang=target_lang,
                pnc="yes"
            )

        end_time = time.time()

        # Cleanup
        os.remove(temp_file_path)
        os.remove(wav_path)

        return JSONResponse({"transcription": transcript, "duration": end_time - start_time})

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"CUDA Error: {str(e)}. Model switched to CPU.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")
