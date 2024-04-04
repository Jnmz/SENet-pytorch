import requests

# The path to the image you want to upload
image_path = 'airplane.jpg'

# The URL of your FastAPI predict endpoint
url = 'http://127.0.0.1:6006/predict/'

# Open the image file in binary mode
with open(image_path, 'rb') as f:
    # Define the files dictionary to include the file in the request
    files = {'file': (image_path, f)}
    # Make the POST request with the file, no need to explicitly set headers as requests does it for multipart/form-data
    response = requests.post(url, files=files)

# Print the response text (JSON output)
print(response.text)
