# ML-Colorize-Webserver

Simple Python web server that accepts a black-n-white image and returns a colorized version, using a trained model.

## Installation

Install [Python](https://www.python.org/downloads/)

Install required libraries

```bash
pip install Flask Pillow torch torchvision
```

## Run

Start webserver

```bash
python app.py
```

Prepare a POST request using postman for example

URL: `http://localhost:5000/colorize`

Body: `form-data`

```bash
Key: image
Type: File (change from text)
Value: Click Select files -> +New file from local machine
```

Select your black-n-white image in the request.

Alternatively, use the prepared image from the folder `img_in`.

# Result
You should see something like this

![instructions](https://github.com/user-attachments/assets/ed9ec3b2-f2ae-4c22-9b49-a069adf24650)
