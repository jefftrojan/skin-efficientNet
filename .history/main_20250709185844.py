import io
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from efficientnet_pytorch import EfficientNet
import uvicorn
from typing import Dict, List

# Initialize FastAPI app
app = FastAPI(
    title="Skin Disease Classification API",
    description="AI-powered skin disease classification with Grad-CAM visualization",
    version="1.0.0"
)

# Configuration
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names from your notebook
CLASS_NAMES = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis',
    'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus',
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer',
    'Sun_Sunlight_Damage', 'Tinea', 'Unknown_Normal', 'Vascular_Tumors',
    'Vasculitis', 'Vitiligo', 'Warts'
]

# Model path - update this to your model file path
MODEL_PATH = "./skin_disease_model.pth"

# Global variables for model and preprocessing
model = None
transform = None
activations = {}
gradients = {}

def load_model():
    """Load the trained EfficientNet model"""
    global model
    try:
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded successfully on {DEVICE}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def setup_transforms():
    """Setup image preprocessing transforms"""
    global transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def forward_hook(module, input, output):
    """Forward hook for Grad-CAM"""
    activations["value"] = output

def backward_hook(module, grad_input, grad_output):
    """Backward hook for Grad-CAM"""
    gradients["value"] = grad_output[0]

def generate_gradcam(model, input_tensor, target_class=None):
    """
    Generate Grad-CAM visualization
    """
    # Clear previous hooks
    activations.clear()
    gradients.clear()
    
    # For EfficientNet, we'll use the last feature layer
    # EfficientNet structure: features -> avgpool -> classifier
    target_layer = model._conv_head  # Last convolutional layer in EfficientNet
    
    # Register hooks
    fwd_hook = target_layer.register_forward_hook(forward_hook)
    bwd_hook = target_layer.register_backward_hook(backward_hook)
    
    try:
        # Forward pass
        model.eval()
        output = model(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        grads = gradients["value"]
        acts = activations["value"]
        
        # Compute weights and cam
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze().cpu().detach().numpy()
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class
        
    finally:
        # Remove hooks
        fwd_hook.remove()
        bwd_hook.remove()

def create_gradcam_overlay(original_img, cam):
    """Create Grad-CAM overlay on original image"""
    # Convert PIL to numpy if needed
    if isinstance(original_img, Image.Image):
        img_array = np.array(original_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    else:
        img_array = original_img
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Convert BGR to RGB for heatmap
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = heatmap * 0.4 + img_array * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay

@app.on_event("startup")
async def startup_event():
    """Initialize model and transforms on startup"""
    setup_transforms()
    if not load_model():
        raise Exception("Failed to load model")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Skin Disease Classification API is running!"}

@app.get("/classes")
async def get_classes():
    """Get all available class names"""
    return {"classes": CLASS_NAMES, "total": len(CLASS_NAMES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin disease class with probabilities
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)
        
        # Prepare all class predictions
        all_predictions = []
        for i, class_name in enumerate(CLASS_NAMES):
            all_predictions.append({
                "class": class_name,
                "probability": float(probabilities[0][i])
            })
        
        # Sort by probability
        all_predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        response = {
            "predicted_class": CLASS_NAMES[top_class.item()],
            "confidence": float(top_prob.item()),
            "all_predictions": all_predictions,
            "image_info": {
                "filename": file.filename,
                "size": image.size,
                "mode": image.mode
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/gradcam")
async def gradcam_visualization(file: UploadFile = File(...)):
    """
    Generate Grad-CAM visualization
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Generate Grad-CAM
        cam, predicted_class = generate_gradcam(model, input_tensor)
        
        # Create overlay
        overlay = create_gradcam_overlay(image, cam)
        
        # Convert to PIL Image and save to bytes
        overlay_img = Image.fromarray(overlay)
        img_byte_arr = io.BytesIO()
        overlay_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Add prediction info to headers
        headers = {
            "X-Predicted-Class": CLASS_NAMES[predicted_class],
            "X-Predicted-Index": str(predicted_class)
        }
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

@app.post("/predict_with_gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Combined endpoint: Get prediction and Grad-CAM in one request
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)
        
        # Generate Grad-CAM
        cam, predicted_class = generate_gradcam(model, input_tensor)
        
        # Create overlay
        overlay = create_gradcam_overlay(image, cam)
        
        # Convert overlay to base64 for JSON response
        overlay_img = Image.fromarray(overlay)
        img_byte_arr = io.BytesIO()
        overlay_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Prepare all class predictions
        all_predictions = []
        for i, class_name in enumerate(CLASS_NAMES):
            all_predictions.append({
                "class": class_name,
                "probability": float(probabilities[0][i])
            })
        
        # Sort by probability
        all_predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        response = {
            "predicted_class": CLASS_NAMES[top_class.item()],
            "confidence": float(top_prob.item()),
            "all_predictions": all_predictions,
            "gradcam_generated": True,
            "image_info": {
                "filename": file.filename,
                "size": image.size,
                "mode": image.mode
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)