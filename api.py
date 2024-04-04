from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort
from PIL import Image
import uvicorn
app = FastAPI()

# 加载ONNX模型
ort_session = ort.InferenceSession("senet.onnx")
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image = image.resize((32, 32))

        if image.mode != "RGB":
            image = image.convert("RGB")
        

        image_array = np.asarray(image)
        

        image_array = image_array / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        inputs = {ort_session.get_inputs()[0].name: image_array}
        outputs = ort_session.run(None, inputs)
        max_index = np.argmax(outputs[0])
        predicted_class = class_names[max_index]

        return {"prediction": predicted_class}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
if __name__ == '__main__':
    uvicorn.run(app="api:app", host="0.0.0.0", port=6006, reload=True)