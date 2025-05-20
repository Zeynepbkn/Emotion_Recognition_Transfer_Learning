# Emotion Recognition Application

This application uses deep learning model that is trained with transfer learning to recognize facial emotions from images. It can identify five different emotions: Angry, Happy, Neutral, Sad, and Surprise.

## Features

- Upload an image or use your webcam to capture a photo
- Instant emotion prediction
- Confidence levels for each emotion class
- Example images for testing
- User-friendly interface powered by Gradio
- Live sharing feature to share your application with others

## Requirements

- Python 3.6+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Emotion_Recognition_Transfer_Learning.git
cd Emotion_Recognition_Transfer_Learning
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset from [OAHEGA Emotion Recognition Dataset](https://data.mendeley.com/datasets/5ck5zz6f2c/1). Download the Angry, Happy, Sad, Surprise, and Neutral classes.

4. Create a folder named "EMOTION RECOGNITION DATASET" in the root directory of the project and place the downloaded emotion class folders inside it.

5. Create a `sample_images` folder in the root directory and add sample images for testing:

   - sample1.jpg - Example of Happy emotion
   - sample2.jpg - Example of Sad emotion
   - sample3.jpg - Example of Angry emotion
   - sample4.jpg - Example of Surprise emotion
   - sample5.jpg - Example of Neutral emotion

   You can use images from the downloaded dataset as samples.

## Usage

1. Run the application:

```bash
python gradioApp.py
```

2. The application will start on your local machine:

```
Running on local URL:  http://127.0.0.1:7860

```

3. Upload an image or take a photo with your webcam

4. Click the "Analyze Emotion" button to get the prediction results

5. View the predicted emotion and confidence levels

## Dataset

The model was trained on the [OAHEGA Emotion Recognition Dataset](https://data.mendeley.com/datasets/5ck5zz6f2c/1), which contains images of different facial emotions. For this project, the following emotion classes were used:

- Angry
- Happy
- Neutral
- Sad
- Surprise

The dataset was downloaded and placed in a folder named "EMOTION RECOGNITION DATASET" within the project directory.

Sample images are available in the `sample_images` directory.

## Model

The application uses a transfer learning approach with a pre-trained model optimized for emotion recognition. The model was trained using transfer learning techniques to leverage knowledge from existing models and adapt it to the emotion recognition task. The trained model is stored in `optimized_emotion_classifier.pkl`.

## Public Sharing

The application now includes a live sharing feature that generates a temporary public URL when you run the app. This URL is valid for 72 hours by default and can be shared with anyone, allowing them to access your application without having to install anything.

To enable this feature, the application is configured with `share=True` in the Gradio launch parameters.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
