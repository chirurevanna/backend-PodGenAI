from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from openai import OpenAI
import os
from pydub import AudioSegment
from fastapi.responses import FileResponse
#from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import tempfile
import numpy as np
from fastapi.responses import JSONResponse
import nemo.collections.asr as nemo_asr
from io import BytesIO
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio.functional as F
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import time
from nemo.collections.asr.models import EncDecRNNTBPEModel

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Check your CUDA installation.")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Is CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")

client = OpenAI(
    api_key="", 
    base_url="https://hoc-lx-gpu02.ad.iem-hoc.de:8080/v1"
    #base_url="https://api.openai.com/v1"
)

'''
def summarize_text(text, custom_prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{custom_prompt}\n\n{text}",
                }
            ],
            #model="gpt-3.5-turbo", 
            model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
             temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

def transcribe_audio(file_path):
    try:
        # Load your original audio file
        audio = AudioSegment.from_mp3(file_path)

        # Determine the length of the audio file in milliseconds
        audio_length = len(audio)

        # Initialize variables for processing
        chunk_length = 60000  # 1 minute in milliseconds
        start = 0
        transcription = ""

        while start < audio_length:
            # Extract a 1-minute chunk from the audio file
            end = min(start + chunk_length, audio_length)
            segment = audio[start:end]

            # Export this clip to a new file
            segment_file_name = f"segment_{start//1000}-{end//1000}.mp3"
            segment.export(segment_file_name, format="mp3")

            # Process the audio segment with the API
            with open(segment_file_name, "rb") as audio_file:
                translation = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )

            # Append the translated text to the transcription
            transcription += translation.text + "\n"

            # Move to the next segment
            start += chunk_length

        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
'''
# Initialize FastAPI app
app = FastAPI()


'''
#processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
#model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")
#model = model.to("cuda")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to("cuda")
model = model.to("cuda")


@app.post("/facebook-wav2vec2")
async def transcribe_facebook(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Convert the audio to WAV format with 16 kHz mono
        audio = AudioSegment.from_file(temp_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_processed_path = f"{temp_file_path}_processed.wav"
        audio.export(temp_processed_path, format="wav")

        # Load audio as a NumPy array
        audio = AudioSegment.from_file(temp_processed_path)
        audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize audio values to the range [-1.0, 1.0]
        audio_samples /= np.iinfo(np.int16).max

        # Break audio into smaller chunks if needed
        chunk_size = 30 * 16000  # 30 seconds per chunk at 16 kHz
        audio_chunks = [
            audio_samples[i:i + chunk_size]
            for i in range(0, len(audio_samples), chunk_size)
        ]

        print(f"Number of chunks: {len(audio_chunks)}")
        print(f"Chunk sizes: {[len(chunk) for chunk in audio_chunks]}")

        print(f"Original audio duration: {len(audio_samples) / 16000} seconds")
        print(f"Processed audio duration: {len(audio) / 1000} seconds")


        # Process each chunk separately
        start_time = time.time()
        transcriptions = []
        for chunk in audio_chunks:
            input_values = processor(
                chunk,
                sampling_rate=16000,  # Ensure the sampling rate matches the model's requirement
                return_tensors="pt",
                padding="longest"
            ).input_values.to("cuda")

            # Predict logits and decode transcription
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            #print(f"Transcription for chunk {i}: {transcription[0]}")
            # Collect the transcription for the current chunk
            transcriptions.append(transcription[0])

        # Cleanup temporary files
        os.remove(temp_file_path)
        os.remove(temp_processed_path)

        # Combine transcriptions from all chunks
        full_transcription = " ".join(transcriptions)
        end_time = time.time()
        # Return transcription as JSON
        return JSONResponse({"transcription": full_transcription, "duration": end_time - start_time})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# FastAPI endpoint for summarizing text from an uploaded file
@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    try:
        # Read and decode the file content
        content = await file.read()
        text = content.decode("utf-8")

        # Define the custom prompt
        custom_prompt = (
            "Imagine you are a professional evaluator of interviews. "
            "You receive the following text as a transcript of the interview and are asked to summarize it. "
            "You do not invent any additional information. There may be errors in the transcript of the interview, "
            "correct them if possible and try to understand the text anyway. "
            "There are adverts in the transcript of the podcast, please ignore them:"
        )

        # Generate the summary
        summary = summarize_text(text, custom_prompt)

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# FastAPI endpoint for transcribing audio files
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Limit to the first 5 minutes of the audio
        audio = AudioSegment.from_mp3(temp_file_path)
        max_duration = 5 * 60 * 1000  # 5 minutes in milliseconds
        trimmed_audio = audio[:max_duration]

        # Save the trimmed audio to a temporary file
        trimmed_file_path = f"trimmed_{file.filename}"
        trimmed_audio.export(trimmed_file_path, format="mp3")

        # Transcribe the trimmed audio file
        transcription = transcribe_audio(trimmed_file_path)

       # Save transcription to a .txt file
        transcription_file_path = f"{file.filename.split('.')[0]}_transcription.txt"
        with open(transcription_file_path, "w") as transcription_file:
            transcription_file.write(transcription)

        # Delete the temporary audio files
        os.remove(temp_file_path)
        os.remove(trimmed_file_path)

        # Return the transcription file as a response
        return FileResponse(
            transcription_file_path,
            media_type="text/plain",
            filename=transcription_file_path
        )

        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

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
        end_time_in_ms = start_time_in_ms + (5 * 60 * 1000)  # Add 10 minutes to start time

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


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"    

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.empty_cache()  # Clear CUDA cache
torch.cuda.synchronize()  # Sync GPU
                
parakeet_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
parakeet_model.to("cuda")

@app.post("/nvidia-parakeet")
async def transcribe_parakeet(file: UploadFile = File(...)):
    try:
        # Save the uploaded MP3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Convert MP3 to WAV (16 kHz mono)
        audio = AudioSegment.from_file(temp_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        chunk_duration_ms = 30 * 1000  # 30 seconds per chunk
        chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

        transcriptions = []
        start_time = time.time()

        for chunk in chunks:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk_path = chunk_file.name
                chunk.export(chunk_path, format="wav")
            
            # Attempt transcription
            try:
                transcription = parakeet_model.transcribe([chunk_path])
                transcriptions.append(transcription[0])
            except RuntimeError as e:
                # Fallback to CPU if GPU fails
                print(f"CUDA error: {e}. Switching to CPU for this chunk.")
                transcription = parakeet_model.transcribe([chunk_path], use_gpu=False)
                transcriptions.append(transcription[0])
            finally:
                os.remove(chunk_path)

        end_time = time.time()
        os.remove(temp_file_path)
        #print(f"Transcriptions: {transcriptions}")
        flat_transcriptions = [item for sublist in transcriptions for item in sublist]

    # Combine the flattened transcriptions into a single string
        combined_transcription = " ".join(flat_transcriptions)
        return JSONResponse({"transcription": combined_transcription, "duration": end_time - start_time})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



'''
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#torch_dtype = torch.float16
model_id = "openai/whisper-medium"
#model_id = "openai/whisper-large-v3-turbo"
#model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
#model = torch.compile(model)
model.to("cuda")


processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30, #30
    batch_size=16, #16
    torch_dtype=torch_dtype,
    device="cuda", 
)


@app.post("/whisper-medium/")
async def transcribe_audio(file: UploadFile):
    """
    Endpoint to transcribe audio using Hugging Face's Whisper model.
    Supports MP3 and WAV files.
    """
    if file.content_type not in ["audio/wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 audio files are supported.")

    try:
        # Read the uploaded file
        audio_bytes = await file.read()

        # Load the audio file as a waveform and sampling rate using torchaudio
        with BytesIO(audio_bytes) as audio_file:
            waveform, sample_rate = torchaudio.load(audio_file)

        # Convert multi-channel audio to mono if necessary
        if waveform.size(0) > 1:  # Check if there are multiple channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample the audio to 16000 Hz (required by Whisper)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        # Convert waveform to numpy for the pipeline
        waveform_np = waveform.squeeze().numpy()
        torch.cuda.empty_cache()  # Clear CUDA cache
        torch.cuda.synchronize()  # Sync GPU
        start_time = time.time()
        # Transcribe audio using the Hugging Face pipeline
        transcription = asr_pipeline({"array": waveform_np, "sampling_rate": 16000})
        end_time = time.time()
        return {"transcription": transcription["text"], "duration": end_time - start_time}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

import assemblyai as aai



@app.post("/assemblyai")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Initialize the transcriber
        transcriber = aai.Transcriber()
        start_time = time.time()
        # Transcribe the audio file
        transcript = transcriber.transcribe(temp_file_path)
        end_time = time.time()
        # Cleanup temporary file
        os.remove(temp_file_path)

        # Return the transcription as JSON
        return JSONResponse({"transcription": transcript.text, "duration": end_time - start_time})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)