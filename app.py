#This interface is used Hugging Face 
import gradio as gr
import numpy as np
from PIL import Image
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from io import BytesIO
import time

# Emotion classes
emotions = []
if os.path.exists('model_classes.txt'):
    with open('model_classes.txt', 'r') as f:
        emotions = [line.strip() for line in f.readlines()]
else:
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]

# FastAI-compatible ResNet34 model definition
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class EmotionResnet34(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionResnet34, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=False)
        )
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.backbone(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

MODEL_PATH = 'emotion_resnet34.safetensors'
model = EmotionResnet34(len(emotions))

try:
    print("Loading model...")
    if os.path.exists(MODEL_PATH):
        try:
            from safetensors.torch import load_file
            print(f"Loading model with SafeTensors: {MODEL_PATH}")
            state_dict = load_file(MODEL_PATH)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Switching to standard ResNet34 model...")
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, len(emotions))
            model.eval()
    else:
        print("Model file not found!")
        print("Switching to standard ResNet34 model...")
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(emotions))
        model.eval()
except Exception as e:
    print(f"Critical error loading model: {e}")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(emotions))
    model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image):
    if image is None:
        return None
    img = Image.fromarray(image).convert('RGB')
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    return tensor

def predict_emotion(image, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        return "Please upload an image", None
    try:
        # Simulate progress for user feedback
        for i in progress.tqdm(range(1), desc="Processing image..."):
            time.sleep(0.1)  # Simulate a short wait for UI feedback
        tensor = preprocess_image(image)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        emotion = emotions[predicted.item()]
        confidence = {emotions[i]: float(probs[i].cpu()) for i in range(len(emotions))}
        return emotion, confidence
    except Exception as e:
        return f"An error occurred: {str(e)}", None

def get_sample_images():
    sample_dir = "sample_images"
    sample_images = []
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_images.append(os.path.join(sample_dir, file))
    return sample_images

custom_css = """
:root {
    --primary-color: #4CAF50;
    --secondary-color: #45a049;
    --background-color: #f8f9fa;
    --text-color: #333;
}
.gradio-container {
    background-color: var(--background-color);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gradio-header {
    background-color: var(--primary-color);
    color: white;
    padding: 1.5rem;
    text-align: center;
    border-radius: 10px 10px 0 0;
    margin-bottom: 1rem;
}
.gradio-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 600;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #777;
}
.emotion-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}
.prediction-box {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 8px;
    background-color: #f0f7ff;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    width: 100%;
}
.result-text {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--primary-color);
}
.primary-button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.7rem 1.5rem;
    border-radius: 5px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s;
}
.primary-button:hover {
    background-color: var(--secondary-color);
}
"""

def create_interface():
    sample_images = get_sample_images()
    with gr.Blocks(title="Emotion Recognition", theme=gr.themes.Soft(), css=custom_css) as app:
        with gr.Row(elem_classes="gradio-header"):
            gr.Markdown("# Facial Emotion Recognition Application")
        gr.Markdown("### Upload an image or use your webcam to analyze your facial emotion.")
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Image Input",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"],
                    height=350,
                    elem_classes="emotion-container"
                )
                predict_btn = gr.Button(
                    "Analyze Emotion",
                    variant="primary",
                    elem_classes="primary-button"
                )
            with gr.Column(scale=1, elem_classes="emotion-container"):
                with gr.Group(elem_classes="prediction-box"):
                    output_emotion = gr.Textbox(
                        label="Predicted Emotion",
                        elem_classes="result-text"
                    )
                    output_confidence = gr.Label(
                        label="Confidence Levels"
                    )
        with gr.Row():
            gr.Markdown("### Sample Images")
        if sample_images:
            gr.Examples(
                examples=sample_images,
                inputs=input_image,
                label="Sample Images - Click to analyze"
            )
        else:
            gr.Markdown(
                """⚠️ No sample images found. Please add sample images to the 'sample_images' directory:
- sample1.jpg (Happy)
- sample2.jpg (Sad)
- sample3.jpg (Angry)
- sample4.jpg (Surprised)
- sample5.jpg (Neutral)
"""
            )
        with gr.Row(elem_classes="footer"):
            gr.Markdown("Developed with Deep Learning and Transfer Learning | Emotion Recognition Dataset")
        predict_btn.click(
            fn=predict_emotion,
            inputs=[input_image],
            outputs=[output_emotion, output_confidence]
        )
        input_image.change(
            fn=lambda: (None, None),
            inputs=None,
            outputs=[output_emotion, output_confidence]
        )
        with gr.Row(elem_classes="accordion-section"):
            with gr.Accordion("How to Use This Application", open=True): # Initially open
                gr.Markdown(
                    """
                    Welcome! Here's a quick guide:
                    **1. Providing an Image:**
                    You have three ways to add a photo to the **"Image Input"** area:
                    *   **From your computer:** Click the **upload icon** (the first icon, an arrow pointing up) located just below the image preview area. This will open your file browser, allowing you to select a photo from your laptop or computer.
                    *   **Using your webcam:** **Click the middle camera icon** (the one with nested circles) below the image preview area. Your webcam will activate and show a live feed within the "Image Input" box. To take a picture, click the new camera icon that appears on your live webcam feed.
                    *   **From an internet URL:** Find an image online and copy its URL. Then, click the clipboard icon (the last icon) below the image preview area. The application will attempt to load the image from the copied URL.
                    **2. Analyzing the Emotion:**
                    Once your chosen image is loaded in the "Image Input" section, click the green **"Analyze Emotion"** button.
                    **3. Viewing the Results:**
                    After a moment, the application will display:
                    *   The **"Predicted Emotion"** in the text box.
                    *   The **"Confidence Levels"** for various emotions below it.
                    These results appear on the right-hand side.
                    *Tip: You can also click on any of the "Sample Images" (if available) to automatically analyze them!*
                    """
                )
    return app

if __name__ == "__main__":
    print("Starting application...")
    app = create_interface()
    app.launch(share=False, show_error=True) 
