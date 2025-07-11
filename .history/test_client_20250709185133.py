import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Health Check:", response.json())

def test_get_classes():
    """Test getting all classes"""
    response = requests.get(f"{BASE_URL}/classes")
    print("Classes:", response.json())

def test_prediction(image_path):
    """Test prediction endpoint"""
    if not Path(image_path).exists():
        print(f"Image file {image_path} not found!")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Top 3 predictions:")
        for i, pred in enumerate(result['all_predictions'][:3]):
            print(f"  {i+1}. {pred['class']}: {pred['probability']:.4f}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_gradcam(image_path, output_path="gradcam_output.png"):
    """Test Grad-CAM endpoint"""
    if not Path(image_path).exists():
        print(f"Image file {image_path} not found!")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/gradcam", files=files)
    
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Grad-CAM saved to {output_path}")
        print(f"Predicted class: {response.headers.get('X-Predicted-Class')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_combined_prediction(image_path):
    """Test combined prediction with Grad-CAM"""
    if not Path(image_path).exists():
        print(f"Image file {image_path} not found!")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/predict_with_gradcam", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Combined Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Grad-CAM generated: {result['gradcam_generated']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Test with your image path
    image_path = "test_image.jpg"  # Replace with your test image path
    
    print("Testing Skin Disease Classification API\n")
    print("=" * 50)
    
    print("1. Health Check")
    test_health_check()
    print()
    
    print("2. Get Classes")
    test_get_classes()
    print()
    
    print("3. Prediction Test")
    test_prediction(image_path)
    print()
    
    print("4. Grad-CAM Test")
    test_gradcam(image_path)
    print()
    
    print("5. Combined Test")
    test_combined_prediction(image_path)
    print()