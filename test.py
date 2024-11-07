from PIL import Image

# Open the image file
image = Image.open("datasets/captured_data/cloudy_noon/segmentation image/seg633.png")

# Convert the image to RGB (if it's not already in RGB mode)
image = image.convert("RGB")

# Get the pixels
pixels = image.getdata()

# Use a set to store unique colors
unique_colors = set(pixels)

# Print all the unique colors (in RGB format)
print("Unique color values (RGB):")
for color in unique_colors:
    print(color)