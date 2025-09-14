from transformers import AutoModel, AutoConfig

# Check DINOv3 model configurations
models_to_check = [
    "facebook/dinov2-small",
    "facebook/dinov2-base", 
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
]

# Try to check DINOv3 models (correct identifiers)
dinov3_models = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m", 
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
]

print("=== DINOv2 Model Dimensions ===")
for model_name in models_to_check:
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"{model_name}: {config.hidden_size} dimensions")
    except Exception as e:
        print(f"{model_name}: Error - {e}")

print("\n=== DINOv3 Model Dimensions ===")
for model_name in dinov3_models:
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"{model_name}: {config.hidden_size} dimensions")
    except Exception as e:
        print(f"{model_name}: Error - {e}")

# Test loading actual model
print("\n=== Testing Model Loading ===")
# Test the correct DINOv3 model
try:
    model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    print(f"DINOv3-vitl16 loaded successfully")
    print(f"Config hidden size: {model.config.hidden_size}")
except Exception as e:
    print(f"DINOv3-vitl16 loading failed: {e}")
    
    # Try the 7B model
    try:
        model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
        print(f"DINOv3-vit7b16 loaded successfully")
        print(f"Config hidden size: {model.config.hidden_size}")
    except Exception as e:
        print(f"DINOv3-vit7b16 loading failed: {e}")
    
    # Fallback to DINOv2-large
    try:
        model = AutoModel.from_pretrained("facebook/dinov2-large")
        print(f"DINOv2-large loaded successfully")
        print(f"Config hidden size: {model.config.hidden_size}")
    except Exception as e:
        print(f"DINOv2-large loading failed: {e}")