from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from typing import Optional
from PIL import Image
import io
import cv2
import tesserocr
import numpy as np
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ngrok
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the LLaMA model and tokenizer
model = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set up ngrok
ngrok_auth_token = "2lBvQQTBJSwgRw2dTqZ1F9vqCAG_4TWPvfo4pzRK4AHkF5tpS"
if not ngrok_auth_token:
    raise ValueError("NGROK_AUTH_TOKEN is not set")

ngrok.set_auth_token(ngrok_auth_token)
listener = ngrok.forward("127.0.0.1:8000", authtoken_from_env=True, domain="bream-dear-physically.ngrok-free.app")

# System message for LLaMA
system_message = (
    """You are a medical report assistant designed to analyze and interpret content from uploaded medical reports. Users may upload an image of a report and ask questions such as, 
    Which readings are high?' or 'Is there anything abnormal?' Your job is to extract data from the report using OCR, identify key values, and determine if any readings exceed standard medical limits. 
    If a user uploads a report without asking a question, provide a friendly, concise short summary highlighting any significant findings. 
    For example, mention elevated values, potential concerns, and simple, actionable recommendations to improve health where necessary. 
    When interpreting readings, correct common OCR errors, such as misinterpreted decimals (e.g., '72' may be '7.2' based on context). 
    Maintain a friendly, supportive tone like a helpful companion, avoiding overly technical language or explanations about decimal corrections. 
    Focus on clear, accurate information that users can easily understand and act upon."""
)

def preprocess_and_ocr(image, extract_digits=False):
    """Preprocess image and perform OCR."""
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(gray_image, -1, kernel)
    blurred_image = cv2.GaussianBlur(sharpened_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_image = Image.fromarray(binary_image)
    api = tesserocr.PyTessBaseAPI(path='/usr/share/tesseract-ocr/4.00/tessdata')
    whitelist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.," if extract_digits else "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,"
    api.SetVariable("tessedit_char_whitelist", whitelist)
    api.SetImage(processed_image)
    text = api.GetUTF8Text()
    api.End()
    return text.strip()

def query_model(prompt, temperature=0.7, max_length=1024):
    """Query the LLaMA model with a given prompt."""
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=pipeline.model.config.pad_token_id,
        eos_token_id=pipeline.model.config.eos_token_id
    )
    return sequences[0]['generated_text']

@app.post('/upload')
async def upload_image(
    request: Request,
    image: UploadFile = File(...)
):
    """Handle image upload and process with or without user query."""
    try:
        # Parse the form data to extract query_text
        form_data = await request.form()
        query_text = form_data.get("query_text", "").strip()

        # Read and process the uploaded image
        image_content = await image.read()
        with Image.open(io.BytesIO(image_content)) as img:
            img = img.convert("RGB")
        extracted_text = preprocess_and_ocr(img, extract_digits=True)

        if query_text:
            # Combine OCR-extracted content with the user query
            prompt = f"{system_message}\nExtracted text: {extracted_text}\nUser question: {query_text} . Answer the question alone don't give additional information"
        else:
            # Summarize the extracted text if no query is provided
            prompt = f"{system_message}\nExtracted text: {extracted_text}. Summarize the report in simple terms, highlight elevated values, and suggest actionable health advice."

        # Generate a response using the LLaMA model
        response = query_model(prompt)
        return JSONResponse(status_code=200, content={"summary": response})

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")
