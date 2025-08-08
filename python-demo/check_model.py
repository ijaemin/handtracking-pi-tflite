import numpy as np
from pathlib import Path

# Load TFLite Interpreter (prefer tflite_runtime)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback


# Locate model and anchor files relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_int8_io_noanchors.tflite"
ANCH_PATH  = SCRIPT_DIR / "anchors_ssd300.npy"

print(f"Loading model: {MODEL_PATH}")
interp = Interpreter(model_path=str(MODEL_PATH))
interp.allocate_tensors()

inp  = interp.get_input_details()[0]
outs = interp.get_output_details()

print("\n[Input]")
print(f" name: {inp['name']}")
print(f" dtype: {inp['dtype']} shape: {inp['shape']}")
print(f" quantization (scale, zero_point): {inp['quantization']}")

print("\n[Outputs]")
for od in outs:
    print(f" name: {od['name']}")
    print(f"  dtype: {od['dtype']} shape: {od['shape']}")
    print(f"  quantization (scale, zero_point): {od['quantization']}")

# Check anchor file existence
anchor_path = ANCH_PATH
if anchor_path.exists():
    anchors = np.load(anchor_path)
    print(f"\n[Anchors] shape: {anchors.shape} dtype: {anchors.dtype} min/max: {float(anchors.min())} {float(anchors.max())}")
else:
    print(f"\n[Anchors] file not found: {anchor_path}")

print("\nModel & anchors check done.")