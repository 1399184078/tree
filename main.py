from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io

app = FastAPI()

@app.post("/process")
async def process_image(file: UploadFile = File(...), method: str = Form("canny")):
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if method == "canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
    elif method == "blur":
        result = cv2.GaussianBlur(img, (15, 15), 0)
    else:
        result = img

    _, buffer = cv2.imencode('.png', result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type='image/png')
