# import pytesseract
# import cv2


# # Assuming only one file is uploaded, get the filename
# filename = r"C:\Users\windows\Downloads\img data\CoVdeMO.jpeg"

# # Read the image using OpenCV
# img = cv2.imread(filename)

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Use pytesseract to extract text
# text = pytesseract.image_to_string(gray_img)

# # Print the extracted text
# print(text)

####################################
# main.py
from fastapi import FastAPI, UploadFile, HTTPException, status
import cv2
import pytesseract
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
  return {"message": "FastAPI application is running"}

@app.post("/extract_text/")
async def extract_text_from_image(image: UploadFile):
    # Validate file type
    # if not image.content_type or not image.content_type.startswith("image/"):
    #     raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Invalid file type. Please upload an image.")

    try:
        # Read the image content
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode the image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if image decoding was successful
        if img is None:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode image.")

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use pytesseract to extract text
        text = pytesseract.image_to_string(gray_img)

        return {"extracted_text": text}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during text extraction: {e}")
    

# To run the app, use the command:
if "__main__" == __name__:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn main:app --reload --host 0.0.0.0 --port 8000