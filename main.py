from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meibomian Gland Grading API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Labels
CLASS_LABELS = {
    0: "Grade 0 - Normal",
    1: "Grade 1 - Mild MGD",
    2: "Grade 2 - Moderate MGD",
    3: "Grade 3 - Severe MGD"
}

GRADE_DESCRIPTIONS = {
    0: "Normal - No visible gland dropout",
    1: "Mild - <25% gland dropout",
    2: "Moderate - 25-75% gland dropout",
    3: "Severe - >75% gland dropout"
}

# Model class (EfficientNet-B3 backbone)
class EnhancedEfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),  # match your checkpoint
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)  # match your checkpoint
        )

    def forward(self, x):
        return self.backbone(x)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedEfficientNetClassifier(num_classes=4)

# Load model
model_path = "model/model_301.pth"  # Adjust if needed
model_loaded = False
model_load_errors = None

try:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    model_loaded = True
    logger.info("✅ Model loaded successfully (EfficientNet-B3)")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model_load_errors = str(e)

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs[0] if outputs.dim() == 2 else outputs.squeeze()
            probabilities = F.softmax(logits, dim=0)
            predicted_class = int(torch.argmax(logits).item())
            confidence = float(probabilities[predicted_class].item())

        return {
            "success": True,
            "predicted_grade": predicted_class,
            "predicted_class": CLASS_LABELS[predicted_class],
            "description": GRADE_DESCRIPTIONS[predicted_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                f"grade_{i}": round(prob.item() * 100, 2)
                for i, prob in enumerate(probabilities)
            },
            "all_grades": [
                {
                    "grade": i,
                    "label": CLASS_LABELS[i],
                    "description": GRADE_DESCRIPTIONS[i],
                    "probability": round(probabilities[i].item() * 100, 2)
                }
                for i in range(4)
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Meibomian Gland Grading API",
        "status": "running",
        "model_loaded": model_loaded,
        "model_load_errors": model_load_errors,
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    if not model_loaded:
        return {"status": "unhealthy", "error": model_load_errors}
    try:
        test_tensor = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            test_output = model(test_tensor)
        return {"status": "healthy", "output_shape": list(test_output.shape)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
