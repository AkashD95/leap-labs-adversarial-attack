import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

# Define the function to load the model and classify images
def classify_mnist(image_path, model_path):
    """
    Classifies an MNIST image using a pre-trained model.
    
    Parameters:
    - image_path: image path to classify
    - model_path (str): path to the pre-trained model file (default: MNIST.pth)
    
    Returns:
    - int, predicted class label
    """
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output
    
    # Define the device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pre-trained model
    # Initialize the model
    model = Net()

# Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Replace with your file path
    model.eval()  # Set the model to evaluation mode

    # Define the transformation for MNIST images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((28, 28)),  # Resize to MNIST image size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize as MNIST expects
    ])

    # Load and transform the image
    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error loading or processing the image: {e}")

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class label
    
    return predicted.item()

# Example usage
if __name__ == "__main__":
    image_file = "path_to_your_image.png"  # Replace with your image file path
    result = classify_mnist(image_file)
    print(f"Predicted class: {result}")