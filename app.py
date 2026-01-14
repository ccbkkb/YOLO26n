from fastapi import FastAPI, UploadFile, File
import uvicorn
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

# 这里的模型文件名要和 Dockerfile/Workflow 里生成的一致
model_path = "model_quantized.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((640, 640))
    img_data = np.array(img).astype('float32') / 255.0
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_tensor = preprocess(contents)
        outputs = session.run(None, {input_name: input_tensor})
        # 返回部分数据证明成功
        return {"status": "success", "data_sample": outputs[0][0].tolist()[:5]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
