import cv2
import numpy as np

def get_2D_bounding_box_from_segmeted_image(segmented_image_path, class_color_map, min_area_boundingbox = 250, dilation_kernel_size = 10):
    # We load the image
    img = cv2.imread(segmented_image_path)

    # We convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Loop through each segmented color that we are intrested in
    boundingboxes_per_class = dict()
    for label in class_color_map.keys():
        color = np.array(class_color_map[label])

        # Create a mask for the current color
        mask = cv2.inRange(img_rgb, color, color)
        
        # To reduce the incontinuities in the mask we will perfomr a dilation
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours for the masked region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours and add them to a list
        boundingboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > min_area_boundingbox:
                boundingboxes.append([x, y, w - dilation_kernel_size, h - dilation_kernel_size])
        
        boundingboxes_per_class[label] = boundingboxes
    
    return boundingboxes_per_class

def draw_rectangles(image_path, bounding_boxes):
    image = cv2.imread(image_path)

    for label in bounding_boxes.keys():
        if len(bounding_boxes[label]) != 0:
            for boundingbox in bounding_boxes[label]:
                x, y, width, height = boundingbox
                cv2.rectangle(image, (x, y), (x + width, y + height), (0,0,225), 2)

    # Display the image with rectangles
    cv2.imshow("Image with Rectangles", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    # Optionally save the modified image
    cv2.imwrite('output_image.jpg', image)



DATA_FOLDER = "captured_data/noon/segmentation image"
class_color_map = {'car': [0, 0, 142], 'motorcycle':[0, 0, 230], 'truck': [0, 0, 70], 'pedestrian': [220, 20, 60], 'traffic signs': [220, 220, 0], 'traffic lights': [250, 170, 30]}
image_path = DATA_FOLDER + "/seg114.png"
bounding_boxes = get_2D_bounding_box_from_segmeted_image(image_path, class_color_map)
draw_rectangles(image_path, bounding_boxes)