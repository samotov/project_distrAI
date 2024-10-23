import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
image_path = 'C:/Users/rehma/Downloads/re.jpg'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def calculate_distance_map(gray_img, max_distance):
    """
    Calculate the distance map from a grayscale depth image, mapping intensities to distances.
    
    Args:
    gray_img (numpy.ndarray): Grayscale image from the depth camera.
    max_distance (float): The maximum measurable distance by the camera in meters.
    
    Returns:
    numpy.ndarray: A distance map where each value represents the distance in meters.
    """
    normalized_gray = gray_img / 255.0  # Normalize grayscale intensities to [0, 1]
    distance_map = max_distance * (1 - normalized_gray)  # Closer objects have lower intensities
    return distance_map

def detect_and_measure_objects(gray_img, distance_map, intensity_threshold=30):
    """
    Detect objects based on intensity and measure their distance in meters.
    
    Args:
    gray_img (numpy.ndarray): Grayscale image from the depth camera.
    distance_map (numpy.ndarray): Precomputed distance map in meters.
    intensity_threshold (int): Threshold to binarize the image for object detection.
    
    Returns:
    list: List of detected objects with their average distance and centroid coordinates.
    """
    _, binary_img = cv2.threshold(gray_img, intensity_threshold, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    object_distances = []
    for label in range(1, num_labels):  # Label 0 is the background
        object_mask = (labels == label)
        object_distance = np.mean(distance_map[object_mask])
        object_distances.append({
            'centroid': centroids[label],
            'bounding_box': stats[label][:4],  # (x, y, width, height)
            'distance': object_distance
        })
    
    return object_distances

def draw_scale(image, max_distance, image_width, image_height):
    """
    Draws scale on the X and Y axes in meters.
    
    Args:
    image (numpy.ndarray): The image to draw the scale on.
    max_distance (float): Maximum distance for the depth camera (in meters).
    image_width (int): The width of the image.
    image_height (int): The height of the image.
    
    Returns:
    numpy.ndarray: The image with the scale drawn.
    """
    scale_color = (0, 255, 255)  # Yellow for scale lines
    step_size_x = int(image_width / 10)
    step_size_y = int(image_height / 10)
    
    for i in range(1, 11):
        # Scale lines on X-axis
        cv2.line(image, (i * step_size_x, 0), (i * step_size_x, image_height), scale_color, 1)
        cv2.putText(image, f"{i * max_distance / 10:.1f}m", (i * step_size_x, image_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, scale_color, 1)
        
        # Scale lines on Y-axis
        cv2.line(image, (0, i * step_size_y), (image_width, i * step_size_y), scale_color, 1)
        cv2.putText(image, f"{i * max_distance / 10:.1f}m", (5, i * step_size_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, scale_color, 1)
    
    return image

# Define the maximum distance (50 meters as per your camera's range)
max_distance = 50.0  # 50 meters

# Step 1: Calculate the distance map (in meters)
distance_map = calculate_distance_map(gray_image, max_distance)

# Step 2: Detect objects and measure their distances
detected_objects = detect_and_measure_objects(gray_image, distance_map)

# Step 3: Create a copy of the original image to mark objects
output_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Step 4: Mark objects by drawing bounding boxes around them
for obj in detected_objects:
    centroid_x, centroid_y = obj['centroid']
    distance = obj['distance']
    x, y, w, h = obj['bounding_box']  # Bounding box coordinates
    
    print(f"Object at distance: {distance:.2f} meters, Coordinates: (x: {int(centroid_x)}, y: {int(centroid_y)}), Bounding Box: (x: {x}, y: {y}, w: {w}, h: {h})")
    
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box
    cv2.putText(output_image, f"{distance:.2f}m", (int(centroid_x), int(centroid_y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Step 5: Add scale to the image (on both X and Y axes)
image_height, image_width = gray_image.shape[:2]
output_image_with_scale = draw_scale(output_image, max_distance, image_width, image_height)

# Step 6: Display the processed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Depth Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_image_with_scale, cv2.COLOR_BGR2RGB))
plt.title('Objects Detected and Marked with Scale')

plt.show()
