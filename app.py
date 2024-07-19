import io
import torch
from torchvision import transforms
from flask import Flask, request, jsonify, send_file
from PIL import Image
from colorize_model import ColorizeModel

# Initialize the Flask app
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Load pre-trained model
model = ColorizeModel()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)  # Ensure the model is moved to the correct device
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
])


def colorize_image(image):
    # Convert image to tensor
    image_tensor = transform(image).unsqueeze(0)

    # Move the image tensor to the device where your model is (likely 'cuda' if using GPU)
    image_tensor = image_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Convert the output tensor to PIL image
    colorized_img = transforms.ToPILImage()(output.squeeze(0).cpu())

    return colorized_img


@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Get the image file from the request
    image_file = request.files['image']

    # Convert to grayscale just in case
    image = Image.open(image_file).convert('L')

    # Colorize the image
    colorized_image = colorize_image(image)

    # Save the colorized image to a BytesIO object
    img_byte_arr = io.BytesIO()
    colorized_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Send the colorized image as a response
    return send_file(img_byte_arr, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
