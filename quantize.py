import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

input_model = "yolo26n.onnx"
output_model = "model_quantized.onnx"

if not os.path.exists(input_model):
    raise FileNotFoundError(f"{input_model} not found!")

print("Starting quantization...")
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QUInt8
)
print(f"Quantization complete: {output_model}")
