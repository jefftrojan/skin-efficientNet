# Skin Disease Classification API

A FastAPI-based web service for classifying skin diseases using a trained EfficientNet model with Grad-CAM visualization.

## Features

- **Classification**: Predict skin disease from 22 different classes
- **Grad-CAM Visualization**: Generate heatmap overlays showing model attention
- **Multiple Endpoints**: Separate and combined prediction/visualization endpoints
- **RESTful API**: Easy to integrate with web applications
- **Docker Support**: Containerized deployment

## Supported Classes

The model can classify the following 22 skin conditions:
- Acne
- Actinic Keratosis
- Benign Tumors
- Bullous
- Candidiasis
- Drug Eruption
- Eczema
- Infestations & Bites
- Lichen
- Lupus
- Moles
- Psoriasis
- Rosacea
- Seborrheic Keratoses
- Skin Cancer
- Sun/Sunlight Damage
- Tinea
- Unknown/Normal
- Vascular Tumors
- Vasculitis
- Vitiligo
- Warts

## Installation

### Prerequisites
- Python 3.9+
- PyTorch
- Your trained model file (`best_efficientnet.pth`)

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd skin-disease-api
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Place your model file:**
   - Copy your trained `best_efficientnet.pth` file to the project root
   - Or update the `MODEL_PATH` variable in `main.py`

4. **Run the application:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Docker Deployment

### Build and run with Docker Compose:
```bash
docker-compose up --build
```

### Or build manually:
```bash
docker build -t skin-disease-api .
docker run -p 8000:8000 -v $(pwd)/best_efficientnet.pth:/app/best_efficientnet.pth skin-disease-api
```

## API Endpoints

### 1. Health Check
```
GET /
```
Returns API status

### 2. Get Classes
```
GET /classes
```
Returns all available skin disease classes

### 3. Predict
```
POST /predict
```
Upload an image and get prediction with probabilities

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "predicted_class": "Acne",
  "confidence": 0.8945,
  "all_predictions": [
    {
      "class": "Acne",
      "probability": 0.8945
    },
    ...
  ],
  "image_info": {
    "filename": "test.jpg",
    "size": [224, 224],
    "mode": "RGB"
  }
}
```

### 4. Grad-CAM
```
POST /gradcam
```
Upload an image and get Grad-CAM visualization

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
- Content-Type: `image/png`
- Body: PNG image with Grad-CAM overlay
- Headers: `X-Predicted-Class`, `X-Predicted-Index`

### 5. Combined Prediction + Grad-CAM
```
POST /predict_with_gradcam
```
Get both prediction and generate Grad-CAM in one request

## Usage Examples

### Using cURL

**Basic prediction:**
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

**Get Grad-CAM:**
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/gradcam --output gradcam.png
```

### Using Python

```python
import requests

# Prediction
with open("test_image.jpg", "rb") as f:
    files = {"file": ("test_image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")

# Grad-CAM
with open("test_image.jpg", "rb") as f:
    files = {"file": ("test_image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/gradcam", files=files)
    with open("gradcam_output.png", "wb") as output:
        output.write(response.content)
```

## Testing

Run the test client:
```bash
python test_client.py
```

Make sure to update the `image_path` variable in the test file with your test image.

## Model Details

- **Architecture**: EfficientNet-B0
- **Input Size**: 224x224 RGB images
- **Classes**: 22 skin disease categories
- **Preprocessing**: Resize, normalize with ImageNet statistics

## API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Error Handling

The API includes comprehensive error handling for:
- Invalid file types
- Missing model files
- Processing errors
- Invalid requests

## Performance Considerations

- **GPU Support**: Automatically uses CUDA if available
- **Model Loading**: Model loaded once on startup
- **Memory Management**: Efficient tensor operations
- **Concurrent Requests**: FastAPI handles multiple requests asynchronously

## File Structure

```
skin-disease-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── test_client.py         # Testing utilities
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── README.md              # This file
└── best_efficientnet.pth  # Your trained model (not included)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue in the repository or contact the maintainers.