# Multi feature evaluation on facial data using CNN

### Overview
This project delivers a multi-feature real-time facial analytics system. It detects eye and mouth states, classifies age, gender, and emotions, and overlays the results on live video. The system integrates MediaPipe’s 468-point 3D Face Mesh for precise landmark extraction with CNN-based models for age, gender, and emotion classification.
Applications range from driver monitoring and fatigue detection to security, surveillance, and adaptive user interfaces.

### Features
Eye and Mouth Detection: Uses geometric ratios (EAR, MAR) for open/closed states.
Age Prediction: Caffe-based deep model (age_deploy.prototxt).
Gender Classification: CNN model (gender_model) for male/female classification.
Emotion Recognition: Custom CNN (face-emotion_model.json) trained to classify 7 emotions (happy, sad, angry, neutral, surprise, disgust, fear).
MediaPipe Landmarks: 468 facial points for robust tracking (see sample visualizations below).
Real-time Streaming: OpenCV integration for live monitoring.

### Modules Used
Computer Vision: mediapipe, opencv-python
Numerical Ops: numpy, math, time
Deep Learning: tensorflow.keras, caffe (for age/gender models)

### Methodology
1. Facial Landmark Detection
MediaPipe Face Mesh (468 points) detects key regions: eyes, mouth, nose, eyebrows and facial outline.
Aspect ratios are computed with Open_Close_Ratios.
2. Eye and Mouth State
detect_eye_mouth_status determines whether eyes/mouth are open or closed in real time.
3. Age, Gender, Emotion Classification
Age: Deep CNN from Caffe (age_deploy.prototxt) predicts age bracket.
Gender: Pre-trained CNN model for male/female classification.
Emotion: Custom CNN (face-emotion_model.json) trained on facial expression datasets.
4. Integration
OpenCV overlays results (age, gender, emotion, eye/mouth state) onto the live video stream.

### Results
Eye/Mouth: Robust to lighting and pose variation.
Gender: ~96% accuracy.
Emotion: ~93% accuracy (happy, sad, angry, neutral, surprise, disgust, fear).
Age: Consistent predictions across age groups.
