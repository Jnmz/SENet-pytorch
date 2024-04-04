import torch
import torch.nn as nn
import onnx
import numpy as np
from net.se_resnet import se_resnet20

model=se_resnet20(num_classes=10, reduction=16)
checkpoint = torch.load("senet.pth")
model.load_state_dict(checkpoint['model_state_dict'])
x = torch.randn(1, 3, 32, 32) 
with torch.no_grad(): 
    model.eval()
    torch.onnx.export( 
        model, 
        x, 
        "senet.onnx", 
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'])