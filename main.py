import requests
import cv2
import numpy as np
import re
import json
from PIL import Image
from io import BytesIO
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    DonutProcessor,
)

# --- Reusable Pre-processing Function ---
def preprocess_image_for_ocr(image_url: str) -> Image.Image:
    """
    Fetches an image from a URL and applies adaptive thresholding to optimize it for OCR.
    Returns a processed PIL Image.
    """
    # response = requests.get(image_url)
    # original_image = Image.open(BytesIO(response.content)).convert("RGB")
    original_image = Image.open(image_url).convert("RGB")
    
    # Convert to OpenCV format (grayscale)
    cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to get a clean black & white image
    processed_cv_image = cv2.adaptiveThreshold(
        cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to a PIL Image
    return Image.fromarray(processed_cv_image)

# --- Model Function 1: Microsoft TrOCR for Raw Text Extraction ---
def extract_text_with_trocr(image_url: str) -> str:
    """
    Performs OCR using TrOCR on a pre-processed image to get a raw text dump.
    Best for unstructured text like paragraphs, signs, or labels.
    """
    try:
        # Get the cleaned, high-contrast image
        processed_pil_image = preprocess_image_for_ocr(image_url)
        processed_pil_image_rgb = processed_pil_image.convert("RGB") 
        
        # Load model and processor
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

        pixel_values = processor(images=processed_pil_image_rgb, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return extracted_text

    except Exception as e:
        return f"An error occurred with TrOCR: {e}"

# --- Model Function 2: Naver Donut for Structured Data Extraction ---
def extract_structured_data_with_donut(image_url: str) -> dict:
    """
    Performs OCR using Donut on a pre-processed image to get structured JSON data.
    Best for documents like receipts, invoices, or forms.
    """
    try:
        # Get the cleaned, high-contrast image
        processed_pil_image = preprocess_image_for_ocr(image_url)
        
        # Donut's processor expects an RGB image
        processed_pil_image_rgb = processed_pil_image.convert("RGB")
        
        # Load model and processor
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = processor(processed_pil_image_rgb, return_tensors="pt").pixel_values
        
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        return processor.token2json(sequence)

    except Exception as e:
        return f"An error occurred with Donut: {e}"

# --- Main Execution Block ---
# This part of the script runs when you execute it directly.
if __name__ == "__main__":
    
    # URL of a structured document (a receipt)
    image_url = 'https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/images/train/train_00001.png'
    
    print(f"Processing image from: {image_url}\n")
    
    # --- Situation 1: You need all text as a single block ---
    print("--- Calling TrOCR for Raw Text Extraction ---")
    raw_text = extract_text_with_trocr(image_url)
    print(raw_text)
    
    print("\n" + "="*50 + "\n")
    
    # --- Situation 2: You need to parse the document into key-value pairs ---
    print("--- Calling Donut for Structured Data Extraction ---")
    structured_data = extract_structured_data_with_donut(image_url)
    # Pretty-print the JSON output
    print(json.dumps(structured_data, indent=4))