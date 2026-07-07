# imaginevisual
# ImagineVisual – AI Text-to-Image Generator

ImagineVisual is a web application that generates images from text prompts using Artificial Intelligence. The application uses the **Hugging Face Inference API** with a text-to-image model to create high-quality images based on user descriptions.

Users simply enter a prompt, and the AI model generates an image that matches the given description.

## Features

* AI-powered text-to-image generation
* Generate images from natural language prompts
* Simple and responsive web interface
* Fast image generation using the Hugging Face Inference API
* Easy-to-use prompt input
* Download generated images (if enabled)

## Technologies Used

* Python
* Flask
* HTML
* CSS
* JavaScript
* Hugging Face Inference API
* FLUX / Stable Diffusion Model

## Project Structure

```text
imaginevisual/
│── app.py
│── templates/
│── requirements.txt
│── start_server.bat
│── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/arfa-mohd/imaginevisual.git
```

Go to the project folder:

```bash
cd imaginevisual
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configure API Key

Create a `.env` file in the project root and add your Hugging Face API key:

```env
HUGGINGFACE_API_TOKEN=your_huggingface_api_key
```

You can get your API key from your Hugging Face account settings.

## Run the Application

```bash
python app.py
```

or

```bash
start_server.bat
```

Open your browser:

```text
http://127.0.0.1:5000
```

## How to Use

1. Open the application.
2. Enter a text prompt describing the image you want.
3. Click **Generate**.
4. Wait a few seconds while the AI model generates the image.
5. View or download the generated image.

## Example Prompt

```text
A futuristic smart city at sunset with flying cars and neon lights.
```

## Future Improvements

* Image history
* Multiple AI models
* Prompt enhancement
* Negative prompts
* Image variations
* User authentication
* Cloud deployment

## Author

**Mohamed Arfath**

B.Tech – Artificial Intelligence and Data Science

GitHub: https://github.com/arfa-mohd
