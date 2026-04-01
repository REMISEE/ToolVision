import base64
import requests

img_path = "/data/home/suchenghao/ToolVision/CodeVision/ocr_test.png"

with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "file": img_b64,
    "fileType": 1
}

resp = requests.post("http://127.0.0.1:8080/ocr", json=payload, timeout=120)
print(resp.status_code)
print(resp.json())