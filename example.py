import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# Set device to MPS
device = torch.device("cpu")

# Load image processor and model
image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

# Move model to the device
model = model.to(device)

# Download and load image
filepath = hf_hub_download(
    repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
)
image = Image.open(filepath).convert("RGB")

# Process image
inputs = image_processor(images=image, return_tensors="pt", do_resize=True, size={"height": 640, "width": 640})

print(inputs["pixel_values"].size())

# Move inputs to the device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get model outputs
outputs = model(**inputs)

# Get logits
logits = outputs.logits  # shape (batch_size, num_labels, height, width)
print(list(logits.shape))

# basically it works on cpu :) mps has some issues with unsupported operations
