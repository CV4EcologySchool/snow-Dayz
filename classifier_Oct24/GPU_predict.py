import torch
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from PIL import Image, ExifTags
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration settings
connection_string = "your_connection_string"  # Azure Blob connection string
container_name = "your_container_name"
model_path = "path_to_your_trained_model.pth"  # Path to the trained model file
output_dir = "predictions"  # Local directory to save predictions
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def load_model(num_of_classes, exp_name, epoch=None, device = 'mps'): ## what does epoch=None do in function? 
    '''
        Creates a model instance and loads the latest model state weights.
    '''

    model_instance = CustomResNet50(num_of_classes)         # create an object instance of our CustomResNet18 class
    # load all model states
    
    state = torch.load(open(f'/Users/catherinebreen/Dropbox/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt', 'rb'), map_location='cpu')  ### what is this doing? 
    model_instance.load_state_dict(state['model'])
    model_instance.to('mps')
    model_instance.eval()

    return model_instance

# Function to check if date is within winter months
def is_winter_month(date):
    return date.month >= 10 or date.month <= 4


def predict(num_of_classes, files, model, device='cpu', batch_size=1):
    with torch.no_grad():
        filenames = []
        predicted_labels = []
        confidences1list = []
        convert_tensor = transforms.ToTensor()
        for i in tqdm.tqdm(range(0, len(files), batch_size)):
            batch_files = files[i:i+batch_size]
            images = []
            batch_filenames = []
            for file in batch_files:
                cameraID = file.split('/')[-2]
                filename = cameraID + '_' + file.split('/')[-1]
                batch_filenames.append(filename)
                img = Image.open(BytesIO(image_data)).convert("RGB")
                img_tensor = convert_tensor(img).unsqueeze(0).to('mps') #device)  # Move image tensor to GPU
                images.append(img_tensor)

            batch_data = torch.cat(images)
            predictions = model(batch_data)
            confidences = torch.nn.Softmax(dim=1)(predictions).cpu().detach().numpy()  # Bring back to CPU if needed
            confidence1 = confidences[:, 1]
            confidences1list.extend(confidence1)
            predict_labels = (confidence1 > 0.5).astype(int)
            predicted_labels.extend(predict_labels)
            filenames.extend(batch_filenames)



# Batch processing variables
predictions = []
batch_size = 1000
batch_count = 0

# Process each image blob in the container
for blob in container_client.list_blobs():
    blob_client = container_client.get_blob_client(blob.name)

    # Get the original timestamp from blob properties or filename
    try:
        # Assuming the timestamp is part of the filename, e.g., "image_20231005.jpg"
        # Adjust the extraction logic based on your actual filenames
        date_str = blob.name.split('_')[1][:8]
        image_date = datetime.strptime(date_str, "%Y%m%d")

        if not is_winter_month(image_date):
            continue  # Skip images not in winter months

        # Download image data as a stream
        stream = blob_client.download_blob()
        image_data = stream.readall()

        # Run prediction
        result = predict(image_data)
        predictions.append({"blob_name": blob.name, "prediction": result, "date": image_date})

        # Save batch every 1000 predictions
        if len(predictions) >= batch_size:
            df = pd.DataFrame(predictions)
            output_path = os.path.join(output_dir, f"predictions_batch_{batch_count}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {output_path}")
            predictions = []  # Reset predictions list for the next batch
            batch_count += 1

    except Exception as e:
        print(f"Error with blob {blob.name}: {e}")

# Save any remaining predictions after the loop
if predictions:
    df = pd.DataFrame(predictions)
    output_path = os.path.join(output_dir, f"predictions_batch_{batch_count}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='predict on deep learning model.')
    parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
    parser.add_argument('--images_folder', help='Path to test_folder', default='test_resized')
    args = parser.parse_args()

    # setup dataloader validation
    files = glob.glob(f"{args.images_folder}/*.JPG")
    print(len(files))

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(device)
    model = load_model(2, args.exp_name, device = 'mps')

    filenames, predicted_labels, confidences = predict(2, files, model)  
    #IPython.embed()
    results = pd.DataFrame({'filename':filenames, 'predicted_labels': predicted_labels, 'confidences': confidences})
    #IPython.embed()
    results.to_csv(f'/Users/catherinebreen/Documents/TEST/scandcam2018_exp_resnet50_2classes_None.csv')
