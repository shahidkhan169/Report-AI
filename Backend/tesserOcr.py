from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
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
#public_url = ngrok.connect(8000, domain="bream-dear-physically.ngrok-free.app")  # This will give you the ngrok public URL
#print(f"FastAPI is live at: {public_url}")

# System message for LLaMA
system_message = (
    "You are a highly specialized AI assistant designed to process and summarize medical and analytical reports based on OCR (Optical Character Recognition) extracted data. Your primary responsibility is to ensure that the extracted text is accurate, meaningful, and presented in a concise, user-friendly format. Pay special attention to numeric data, as OCR may misinterpret decimal values (e.g., extracting '7.2' as '72'). Correct such errors by cross-referencing realistic and medically appropriate ranges. For instance, blood glucose levels should typically range between 70–140 mg/dL, and systolic/diastolic blood pressure values should fall within standard limits. Always verify and contextualize the extracted text, ensuring critical information is not lost or misrepresented. When summarizing reports, focus on clarity and precision, highlighting key findings, anomalies, and actionable insights. If the extracted data is ambiguous, provide the best possible interpretation based on the context and flag areas of uncertainty for further review. Your goal is to deliver accurate and actionable summaries that help users make informed decisions while ensuring the integrity of the original report's data."
)

def preprocess_and_ocr(image, extract_digits=False):
    # Convert to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Apply sharpening filter
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpen filter
    sharpened_image = cv2.filter2D(gray_image, -1, kernel)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(sharpened_image, (5, 5), 0)

    # Threshold the image to get binary (black and white) image
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert OpenCV image to PIL format for tesserocr
    processed_image = Image.fromarray(binary_image)

    # Configure tesserocr to recognize both text and decimal points
    api = tesserocr.PyTessBaseAPI(path='/usr/share/tesseract-ocr/4.00/tessdata')
    if extract_digits:
        api.SetVariable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,")
    else:
        api.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,")
    
    # Perform OCR
    api.SetImage(processed_image)
    text = api.GetUTF8Text()
    api.End()

    return text

def query_model(prompt, temperature=0.7, max_length=1024):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=pipeline.model.config.pad_token_id
    )
    return sequences[0]['generated_text']

@app.post('/upload')
async def upload_image(image: UploadFile = File(...)):
    try:
        # Read image content
        image_content = await image.read()

        # Open and preprocess the image
        with Image.open(io.BytesIO(image_content)) as img:
            img = img.convert("RGB")

        extracted_text = preprocess_and_ocr(img, extract_digits=True)

        # Prepare query for LLaMA
        query = f"Extracted text: {extracted_text}. Summarize this report, correcting any numeric errors as needed."

        # Generate summary using LLaMA
        response = query_model(f"{system_message}\n{query}")

        return JSONResponse(status_code=200, content={"summary": response})
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
