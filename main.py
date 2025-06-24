from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import requests

app = FastAPI()

# 上传图像文件方式
@app.post("/process")
async def process_image(file: UploadFile = File(...), method: str = Form("canny")):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result = process_core(img, method)
    _, buffer = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

# URL 下载图像方式
@app.post("/process_url")
async def process_from_url(url: str = Form(...), method: str = Form("canny")):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"下载图像失败: {e}")

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="提供的 URL 不是图像")

    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result = process_core(img, method)
    _, buffer = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

# 图像处理核心逻辑
def process_core(img, method):
    if method == "canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    elif method == "blur":
        return cv2.GaussianBlur(img, (15, 15), 0)
    else:
        return img
