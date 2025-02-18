from fastapi import FastAPI, File, UploadFile, Form
from openai import OpenAI
from transformers import pipeline
from fastapi.responses import Response
import torch
app = FastAPI()
import os


from google import genai

# Initialize Gemini Flash Client
gemini_client= genai.Client(api_key="")


# Initialize OpenAI GPT Client
openai_client = OpenAI(api_key="")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.empty_cache()  # Clear CUDA cache
torch.cuda.synchronize()  # Sync GPU
# Initialize LLaMA Model
model_id = "meta-llama/Llama-3.2-1B"  # Change if using a different version
llama_pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16,
    #device_map="auto",
    device="cpu"
)




async def read_text_file(file: UploadFile) -> str:
    """Helper function to read text file contents."""
    content = await file.read()
    return content.decode("utf-8")


@app.post("/summarize/gemini")
async def summarize_with_gemini(prompt: str = Form(...), file: UploadFile = File(...)):
    """Summarize the uploaded transcript file using Gemini Flash 2.0."""
    text = await read_text_file(file)

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{prompt}\n\n{text}"
    )

    summary = f"### Summary by Gemini Flash\n\n{response.text}"
    return Response(content=summary, media_type="text/markdown")

@app.post("/summarize/gpt")
async def summarize_with_gpt(prompt: str = Form(...), file: UploadFile = File(...)):
    """Summarize the uploaded transcript file using GPT-4o."""
    text = await read_text_file(file)

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"{prompt}\n\n{text}"}
        ]
    )

    summary = completion.choices[0].message.content
    return Response(content=summary, media_type="text/markdown")
    

@app.post("/summarize/llama", response_class=Response)
async def summarize_with_llama(prompt: str = Form(...), file: UploadFile = File(...)):
    """Summarize using LLaMA model with large input handling."""
    text = await read_text_file(file)
    
    # LLaMA's input size handling - truncating if exceeds token limits
    input_text = f"{prompt}\n\n{text}"

    result = llama_pipe(
        input_text,
        max_new_tokens=1000,  # Can increase if needed
        do_sample=True,
        temperature=0.7,  # Adjust for randomness
        top_k=50,  # Helps control diversity
        top_p=0.95,  # Nucleus sampling
        batch_size=1
    )

    generated_text = result[0]["generated_text"]

    summary = f"### Summary by LLaMA\n\n{generated_text}"
    return Response(content=summary, media_type="text/markdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
