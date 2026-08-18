import onnx

# Target all three models to ensure the entire edge pipeline stays in sync
models_to_patch = [
    "inference/anomaly_detector.onnx",
    "inference/model2_cifar10.onnx",
    "inference/model3_lstm.onnx"
]

for model_path in models_to_patch:
    print(f"\n--- Processing: {model_path} ---")
    try:
        print(f"Loading {model_path}...")
        model = onnx.load(model_path)

        # 1. Force the IR Version
        model.ir_version = 9
        print("-> Forced IR Version to 9")

        # 2. Force the Opset Imports
        for imp in model.opset_import:
            # Fix machine learning opsets
            if imp.domain == 'ai.onnx.ml' and imp.version > 3:
                print(f"-> Downgraded 'ai.onnx.ml' opset from {imp.version} to 3")
                imp.version = 3

            # Fix the standard opsets (Catches opset 18 from the LSTM)
            if imp.domain == '' and imp.version > 14:
                print(f"-> Downgraded default opset from {imp.version} to 14")
                imp.version = 14

        # Overwrite the file
        onnx.save(model, model_path)
        print(f"STATUS: [SUCCESS] Model metadata fully patched for ARM compatibility: {model_path}")

    except Exception as e:
        print(f"STATUS: [FAILED] Could not patch {model_path}. Error: {e}")