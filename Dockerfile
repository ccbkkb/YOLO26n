FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 App 代码
COPY app.py .

# 关键：这里假设构建环境已经有了转换好的模型
# 我们稍后在 GitHub Action 中会把模型生成在当前目录
COPY model_quantized.onnx .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
