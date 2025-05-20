import pickle
import gradio as gr
import numpy as np
from PIL import Image
import os
import time
try:
    from fastai.learner import load_learner
    fastai_available = True
except ImportError:
    fastai_available = False
    print("FastAI is not installed. Please install it using: pip install fastai")

# Emotion classes
emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]  # Model's predicted classes

# Load the model
try:
    print("Loading the model...")
    if os.path.exists('optimized_emotion_classifier.pkl'):
        if fastai_available:
            try:
                print("Trying to load the model with FastAI...")
                model = load_learner('optimized_emotion_classifier.pkl')
                print("Model loaded successfully with FastAI!")
            except Exception as e:
                print(f"Error occurred while loading the model with FastAI: {e}")
                # Try to load as pickle file as a fallback
                with open('optimized_emotion_classifier.pkl', 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded successfully as pickle file!")
        else:
            print("FastAI library is not installed. Please install it using: pip install fastai")
            raise ImportError("FastAI library is not installed.")
    else:
        print("Model file not found!")
        raise FileNotFoundError("optimized_emotion_classifier.pkl file not found.")
except Exception as e:
    print(f"Critical error occurred while loading the model: {e}")
    raise

# Preprocess the image
def preprocess_image(image):
    if image is None:
        return None
    
    # Resize the image to the model's expected size (224x224 for ResNet models)
    img = Image.fromarray(image).resize((224, 224))
    
    # Return the PIL image for FastAI model
    return img

# Predict the emotion
def predict_emotion(image):
    if image is None:
        return "Please upload an image or take a photo", None
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # FastAI model prediction
        prediction = model.predict(processed_image)
        
        # FastAI prediction[0] returns the class name, prediction[2] returns the probabilities
        emotion = prediction[0]
        
        # All emotions probabilities
        probs = prediction[2].numpy()
        confidence = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
        
        return emotion, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Error occurred: {str(e)}", None

# Capture photo and predict emotion function
def capture_and_predict(image_input):
    # If webcam is open, capture photo and predict
    if image_input is not None:
        return predict_emotion(image_input)
    else:
        return "Camera is not open or photo is not taken", None

# Get sample images path
def get_sample_images():
    sample_images = []
    sample_dir = "sample_images"
    
    # Create sample_images directory if it doesn't exist
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"Created {sample_dir} directory")
        return []
    
    # Get all jpg and png files from sample_images directory
    for file in os.listdir(sample_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_images.append(os.path.join(sample_dir, file))
    
    # If no sample images are available, try to get some from the dataset
    if not sample_images:
        try:
            dataset_dir = "EMOTION RECOGNITION DATASET"
            for emotion in emotions:
                emotion_dir = os.path.join(dataset_dir, emotion)
                if os.path.exists(emotion_dir):
                    for i, file in enumerate(os.listdir(emotion_dir)):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')) and i < 1:
                            target_path = os.path.join(sample_dir, f"sample_{emotion.lower()}.jpg")
                            # Copy the file to sample_images directory
                            with open(os.path.join(emotion_dir, file), 'rb') as src:
                                with open(target_path, 'wb') as dst:
                                    dst.write(src.read())
                            sample_images.append(target_path)
        except Exception as e:
            print(f"Could not create sample images from dataset: {e}")
    
    return sample_images

# Custom CSS for a more professional look
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

# Create Gradio interface with a professional look
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

# Main application
if __name__ == "__main__":
    print("Application starting...")
    app = create_interface()
    # Launch with both local 
    app.launch(share=False)
    
    
