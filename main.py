import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:/Users/rehma/Downloads/image2.png'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


max_distance = 50.0

def calculate_distance(intensity, max_distance):
    """Map grayscale intensity (0-255) to a distance value."""
    normalized_intensity = intensity / 255.0  # Normalize to [0, 1]
    distance = max_distance * (1 - normalized_intensity)  # Invert scale distance
    return distance

# Use edge detection to help identify the bounding boxes
edges = cv2.Canny(gray_image, threshold1=100, threshold2=30)

# Find contours based on the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to keep only those that look like bounding boxes
bounding_boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 10:  # Filter out very small boxes
        bounding_boxes.append((x, y, w, h))

# Calculate distances for each detected bounding box
detected_boxes_info = []
for (x, y, w, h) in bounding_boxes:
    # Extract Region of Interest (ROI) within the bounding box
    object_roi_gray = gray_image[y:y+h, x:x+w]
    
    # Calculate the mean intensity in the bounding box
    mean_intensity = np.mean(object_roi_gray)
    mean_distance = calculate_distance(mean_intensity, max_distance)
    
    # Store each detected bounding box's information
    detected_boxes_info.append({
        'bounding_box': (x, y, w, h),
        'mean_intensity': mean_intensity,
        'mean_distance': mean_distance
    })
    
    # Print the result for each bounding box
    print(f"Bounding Box: ({x}, {y}, {w}, {h}), Mean Distance: {mean_distance:.2f} meters, Mean Intensity: {mean_intensity}")

# Display bounding boxes with distances on the image
output_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

for box_info in detected_boxes_info:
    x, y, w, h = box_info['bounding_box']
    mean_distance = box_info['mean_distance']
    
    # Draw the bounding box and distance on the image
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(output_image, f"{mean_distance:.2f}m", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the image with bounding boxes and distances
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Bounding Boxes with Distances')
plt.axis('off')
plt.show()
