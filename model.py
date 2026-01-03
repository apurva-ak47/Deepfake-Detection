from tensorflow.keras.models import load_model
import os

if os.path.exists("deepfake_model.h5"):
    model = load_model("deepfake_model.h5")
    print("✅ Loaded trained deepfake model")
else:
    raise FileNotFoundError("❌ deepfake_model.h5 not found — please train the model first.")
