# SENet-pytorch
Implementing **SENet** using the pytorch framework

## requirement
* python3.8
* pytorch1.6
* torchvision
* tensorboard
* tqdm
* fastapi
* uvicorn
* onnxruntime

## dataset
The data set is automatically downloaded to the data folder, if the speed is too slow,You can try the following link
https://share.weiyun.com/56FKfYz
password：nwdmtc

## train
```
python train.py
```
If you want to train a model without SE block
```
python train.py --baseline
```
Trainers can change parameters in train.py according to their needs.

## log
The log file will be saved in the logs folder
```
 tensorboard --logdir=logs
```
You can view the training process and results

## some result

|model             | ResNet20       | SE-ResNet20    |
|:-------------    | :------------- | :------------- |
|max  val accuracy |  92.2%           | 92.5%          |

## PyTorch Model to ONNX Conversion
This repository now supports converting PyTorch models to the ONNX format for broader compatibility with inference engines.
```
 python pt2onnx.py
```

## Efficient Inference with ONNX Runtime and FastAPI
For efficient model inference, we leverage ONNX Runtime within a FastAPI application. This setup allows for rapid deployment and high-performance predictions.

## Running the FastAPI Inference Server
After converting your model to ONNX, you can serve it with FastAPI:
```
 python api.py
```
This command starts the FastAPI server, making your model accessible for inference via REST API calls.
For testing the FastAPI server with a POST request, you can use the `post.py` script.
```
 python post.py
```
This script makes a POST request to the /predict endpoint of your FastAPI application with a JSON payload. Make sure to modify the url and data variables to fit your application's needs.
