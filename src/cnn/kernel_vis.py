"""
Visualize kernels applied in CNN - 1st iteration.
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_model import ObservationProbabilityModel
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
file_path = os.path.join(parent_dir, 'data', 'observation_model.pth')
sample_img_path = os.path.join(parent_dir, 'data', 'ENTER THE NAME OF YOUR IMAGE') # ADD AN IMAGE NAME

conv_net = ObservationProbabilityModel()
conv_net.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
conv_net.eval()

# Get the weights of the first convolutional layer of the network
first_conv_layer = conv_net.cnn[0]
kernels = first_conv_layer.weight.data.cpu().numpy()

# Visualize kernels from first convolutional layer
kernels_normalized = kernels.copy()
for i in range(kernels_normalized.shape[0]):
    # Note: shape is (out_channels, in_channels, height, width)
    # For RGB images, in_channels = 3
    for c in range(kernels_normalized.shape[1]):
        kernel = kernels_normalized[i, c]
        kernel_min = kernel.min()
        kernel_max = kernel.max()
        if kernel_max - kernel_min > 0:
            kernels_normalized[i, c] = (kernel - kernel_min) / (kernel_max - kernel_min)


num_kernels = kernels.shape[0]
grid_size = int(np.ceil(np.sqrt(num_kernels)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
fig.suptitle('Kernels Learned at First Convolutional Layer', fontsize=16, fontweight='bold')

for i in range(grid_size*grid_size):
    row = i // grid_size
    col = i % grid_size
    if i < num_kernels:
        # Transpose to (height, width, channels) for RGB display
        kernel_rgb = np.transpose(kernels_normalized[i], (1, 2, 0))
        axes[row, col].imshow(kernel_rgb)
        axes[row, col].set_title(f'K{i}', fontsize=8)
    else:
        axes[row, col].axis('off')
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.tight_layout()
plt.savefig('kernel_grid.png', dpi=150, bbox_inches='tight')
plt.close()

# Apply kernels to sample image
img = cv2.imread(sample_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img = cv2.resize(img, (256, 256))  # Match IMAGE_SIZE from training
img = img / 255.0 # Normalize the image

# Apply same normalization as training
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = (img - mean) / std

# Convert to tensor: (H, W, C) -> (C, H, W)
img_tensor = torch.tensor(img).float().permute(2, 0, 1)
img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
print(f"Image tensor shape: {img_tensor.shape}")

# Apply the first conv layer to the image
with torch.no_grad():
    output = first_conv_layer(img_tensor)


output = output.squeeze(0) # Remove batch dimension
output_np = output.cpu().numpy()

# Normalize each channel for visualization
output_normalized = np.zeros_like(output_np)
for i in range(output_np.shape[0]):
    channel = output_np[i]
    channel_min = channel.min()
    channel_max = channel.max()
    if channel_max - channel_min > 0:
        output_normalized[i] = (channel - channel_min) / (channel_max - channel_min)

# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
num_outputs = output_normalized.shape[0]
grid_size = int(np.ceil(np.sqrt(num_outputs)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
fig.suptitle('Feature Maps from First Convolutional Layer', fontsize=16, fontweight='bold')

for i in range(grid_size * grid_size):
    row = i // grid_size
    col = i % grid_size
    if i < num_outputs:
        axes[row, col].imshow(output_normalized[i], cmap='gray')
        axes[row, col].set_title(f'Kernel {i}', fontsize=8)
    else:
        axes[row, col].axis('off')   
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.
plt.tight_layout()
plt.savefig('image_transform_grid.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature map progression through layers
# Create a feature map progression. You can manually specify the forward pass order or programatically track each activation through the forward pass of the CNN.
with torch.no_grad():
    x = img_tensor

    # Layer 0-2: Conv1, BatchNorm, ReLU
    x1_conv = conv_net.cnn[0](x)
    x1_bn = conv_net.cnn[1](x1_conv)
    x1_relu = conv_net.cnn[2](x1_bn)
    x1_pool = conv_net.cnn[3](x1_relu)  # MaxPool

    # Layer 4-6: Conv2, BatchNorm, ReLU
    x2_conv = conv_net.cnn[4](x1_pool)
    x2_bn = conv_net.cnn[5](x2_conv)
    x2_relu = conv_net.cnn[6](x2_bn)
    x2_pool = conv_net.cnn[7](x2_relu)  # MaxPool

    # Layer 8-10: Conv3, BatchNorm, ReLU
    x3_conv = conv_net.cnn[8](x2_pool)
    x3_bn = conv_net.cnn[9](x3_conv)
    x3_relu = conv_net.cnn[10](x3_bn)
    x3_pool = conv_net.cnn[11](x3_relu)  # MaxPool

# Prepare original image for display (undo normalization)
orig_img = img_tensor.squeeze().cpu().numpy()
orig_img = np.transpose(orig_img, (1, 2, 0))  # C, H, W -> H, W, C
orig_img = orig_img * std + mean  # Denormalize
orig_img = np.clip(orig_img, 0, 1)

# Show first channel of each layer
layers = [
    ('Original Image', orig_img),
    ('Conv1 Output', x1_conv[0, 0].cpu().numpy()),
    ('Conv1 + BN + ReLU', x1_relu[0, 0].cpu().numpy()),
    ('Conv1 + Pool', x1_pool[0, 0].cpu().numpy()),
    ('Conv2 Output', x2_conv[0, 0].cpu().numpy()),
    ('Conv2 + BN + ReLU', x2_relu[0, 0].cpu().numpy()),
    ('Conv2 + Pool', x2_pool[0, 0].cpu().numpy()),
    ('Conv3 Output', x3_conv[0, 0].cpu().numpy()),
    ('Conv3 + BN + ReLU', x3_relu[0, 0].cpu().numpy()),
    ('Conv3 + Pool', x3_pool[0, 0].cpu().numpy()),
]

# Plot feature progression
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Feature Map Progression Through CNN Layers (First Channel Only)', 
             fontsize=16, fontweight='bold')

for idx, (title, feature_map) in enumerate(layers):
    row = idx // 5
    col = idx % 5
    
    fmap = feature_map.copy() if idx > 0 else feature_map
    
    # Normalize for visualization (except original which is already normalized)
    if idx > 0:
        fmap_min = fmap.min()
        fmap_max = fmap.max()
        if fmap_max - fmap_min > 0:
            fmap = (fmap - fmap_min) / (fmap_max - fmap_min)
        axes[row, col].imshow(fmap, cmap='viridis')
    else:
        axes[row, col].imshow(fmap) # RGB image
    
    axes[row, col].set_title(title, fontsize=10, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('feature_progression.png', dpi=150, bbox_inches='tight')
plt.close()
