# src/model_runtime.py
import os
import logging

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# These are imported by prediction_routes.py
model = None
model_loaded = False
device = None
transform = None

# Update these to match your project/classes
CLASS_LABELS = ["Grade 0", "Grade 1", "Grade 2", "Grade 3"]
GRADE_DESCRIPTIONS = {
    0: "No DR / Normal",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
}

def load_model_once():
    """
    Loads the PyTorch model + device + transforms one time.
    Safe to call on every request (it will only load once).
    """
    global model, model_loaded, device, transform

    if model_loaded:
        return

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---- Image preprocessing transform (must match training) ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # change if your model expects different size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---- Load model weights ----
    # Put your model weights file in a known place.
    # Example: eye-ai-backend/flask/models/model.pth
    model_path = os.environ.get(
    "MODEL_PATH",
    "C:/Users/kumar/eye-ai-backend/model/model_301.pth"
)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MODEL_PATH not found: {model_path}. "
            f"Set env var MODEL_PATH or place weights at C:/Users/kumar/eye-ai-backend/model/model_301.pth"
        )

    # IMPORTANT: You must create the model architecture here
    # (or load a TorchScript model).
    #
    # Option 1 (TorchScript) - easiest to deploy:
    # model = torch.jit.load(model_path, map_location=device)
    #
    # Option 2 (state_dict) - requires your model class definition:
    # from src.your_model_def import MyModel
    # model = MyModel(...)
    # state = torch.load(model_path, map_location=device)
    # model.load_state_dict(state)

    # ---- Pick ONE method below ----
    # Method A: TorchScript (recommended if you have .pt / torchscript)
    model = torch.jit.load(model_path, map_location=device)

    model.eval()
    model_loaded = True
    logger.info("Model loaded successfully")