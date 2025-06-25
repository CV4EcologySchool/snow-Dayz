import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from model import get_model

def predict_and_save(model, image_path, output_path, transform=None, device='cpu'):
    model.eval()
    model.to(device)

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
    
    snow_depth = output.item()
    
    # Optional: assume confidence is inverse of loss uncertainty
    # This is just a placeholder; proper confidence estimation requires MC dropout or ensembling
    confidence = 1.0  # placeholder confidence

    # Annotate image
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)

    try:
        font = ImageFont.truetype("arialbd.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    text = f"Snow Depth: {snow_depth:.2f} cm, Conf: {confidence:.2f}"
    draw.text((10, 10), text, fill="white", font=font)

    # Save annotated image
    annotated_img.save(output_path)
    print(f"Saved image with prediction: {output_path}")

# Define same transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load model
model = get_model()
model.load_state_dict(torch.load("/Users/catherinebreen/code/Chapter1/snowdepth_vit.pt", map_location="cpu"))

# Predict on a new image
predict_and_save(
    model,
    image_path="/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/CHE8/CHE8_IMG_0007.JPG",
    output_path="test_image_predicted.jpg",
    transform=transform,
    device='cpu'
)
