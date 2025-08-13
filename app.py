
# Import required libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import gradio as gr
from uniface import RetinaFace
from huggingface_hub import hf_hub_download

# Step 1: Define image transformations for test/validation data
transform_test = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL image
    transforms.Resize((128, 128)),  # Resize image to 128x128
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image
])

# Step 2: Define class names for FER2013 dataset
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Step 3: Load RetinaFace model for face detection
retinaface = RetinaFace(
    model_name="retinaface_r34",  # Use ResNet34-based RetinaFace model
    conf_thresh=0.3,         # Confidence threshold for face detection
    nms_thresh=0.3,          # Non-maximum suppression threshold
    input_size=1024,         # Input size for face detector
    dynamic_size=True        # Allow dynamic input size
)

# Step 4: Load VGG19 model architecture and adjust for FER2013
model = models.vgg19(weights=None)
model.classifier[5] = nn.Dropout(0.5)  # Increase regularization
model.classifier[6] = nn.Linear(4096, 7)  # FER2013 has 7 classes

# Step 5: Load model weights from HuggingFace Hub
model_path = hf_hub_download(repo_id="hanseltertius/vgg19-finetuned-fer2013", filename="best_fer32_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Step 6: Define the emotion detection function for Gradio
def detect_emotion(image):
    # Detect faces in the input image using RetinaFace
    faces = retinaface.detect(image)
    if faces is None or len(faces[0]) == 0:
        return image, {"error": "No face detected"}
    results = []
    boxes = faces[0]
    annotated_img = image.copy()  # Copy image for annotation
    for box in boxes:
        # Extract bounding box coordinates and score
        x1, y1, x2, y2, _ = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4])
        face_img = image[y1:y2, x1:x2]  # Crop face from image
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_img_resized = cv2.resize(face_img_rgb, (128, 128))  # Resize to model input size
        face_tensor = transform_test(face_img_resized).unsqueeze(0).to(device)  # Preprocess and add batch dimension
        with torch.no_grad():  # Disable gradients for inference
            outputs = model(face_tensor)  # Get model predictions
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            conf, pred = torch.max(probs, 1)  # Get highest probability and predicted class
            emotion = class_names[pred.item()]  # Get emotion label
            confidence = round(conf.item() * 100, 2)  # Convert confidence to percentage
        # Draw bounding box and label on annotated image
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, f"{emotion} ({confidence}%)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })
    return annotated_img, {"result": results}

# Step 7: Create Gradio interface for emotion detection
iface = gr.Interface(
    fn=detect_emotion,  # Function to run for each input
    inputs=gr.Image(type="numpy", label="Upload Image"),  # Input: image upload
    outputs=[gr.Image(type="numpy", label="Annotated Image"), gr.JSON(label="Detection Results")],  # Output: annotated image and JSON results
    title="Facial Emotion Recognition with RetinaFace & VGG19 Pretrained Model",
    description="Upload an image. The app will detect all human faces using RetinaFace powered by uniface library, which predicts the emotion and confidence for each face using VGG19 model trained on FER2013 Dataset from HuggingFace. Returns annotated image and JSON with emotion, confidence, and bounding box coordinates."
)

# Step 8: Launch Gradio app
if __name__ == "__main__":
    iface.launch(share=True)
