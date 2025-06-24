from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import cv2
import io
import requests
import base64
from fastapi.responses import JSONResponse


app = FastAPI()

# 上传图像文件方式（保持不变）
@app.post("/process")
async def process_image(file: UploadFile = File(...), method: str = "canny"):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图像")

    result = process_core(img, method)
    _, buffer = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

# JSON 请求体版本的 URL 下载接口
class UrlRequest(BaseModel):
    url: str
    method: str = "canny"

@app.post("/process_url")
async def process_from_url(req: UrlRequest):
    try:
        response = requests.get(req.url, timeout=5)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"下载图像失败: {e}")

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="提供的 URL 不是图像")

    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图像")

    result = process_core(img, req.method)
    _, buffer = cv2.imencode(".png", result)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return JSONResponse(content={"image_base64": image_base64})

def process_core(img, method):
    method = method.lower()

    if method == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif method == "binary":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

    elif method == "mean":
        return cv2.blur(img, (5, 5))

    elif method == "median":
        return cv2.medianBlur(img, 5)

    elif method == "canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)

    elif method == "equalize":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    elif method == "segment":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return segmented

    else:
        return img

