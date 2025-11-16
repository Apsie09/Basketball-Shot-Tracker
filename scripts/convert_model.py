#!/usr/bin/env python3
"""Convert PyTorch model to ONNX format"""

import torch
from pathlib import Path

def convert_to_onnx(pt_model_path, onnx_output_path, img_size=640):
    """
    Convert PyTorch YOLOv5/v8 model to ONNX
    
    Args:
        pt_model_path: Path to .pt model
        onnx_output_path: Output path for .onnx
        img_size: Input image size
    """
    # Load model
    model = torch.load(pt_model_path, map_location='cpu')
    model = model['model'] if isinstance(model, dict) else model
    
    # Set to evaluation mode
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
    
    print(f"Model converted successfully: {onnx_output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_model.py <input.pt> <output.onnx>")
        sys.exit(1)
    
    convert_to_onnx(sys.argv[1], sys.argv[2])