import cv2
import os
import numpy as np

# Parameters for drawing
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial x, y coordinates of the region
annotations = []  # List to store segmentation points
bbox_annotations = []  # List to store bounding box info

# Mouse callback function to draw contours and record bounding boxes
def draw_contour(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y  # Starting point of the rectangle
        annotations.append([(x, y)])  # Start a new contour

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Add points to the current contour
            annotations[-1].append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        annotations[-1].append((x, y))  # Close the contour
        # Calculate bounding box
        x_min = min(ix, x)
        y_min = min(iy, y)
        x_max = max(ix, x)
        y_max = max(iy, y)
        w = x_max - x_min
        h = y_max - y_min
        bbox_annotations.append((x_min, y_min, w, h))
        print(f"Bounding box: x={x_min}, y={y_min}, w={w}, h={h}")

# Function to display the image and collect annotations
def segment_image(image_path):
    global annotations, bbox_annotations

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    # Create a clone of the image for annotation display
    annotated_image = image.copy()
    cv2.namedWindow("Image Segmentation")
    cv2.setMouseCallback("Image Segmentation", draw_contour)

    annotations = []  # Reset annotations for this image
    bbox_annotations = []  # Reset bounding boxes

    while True:
        # Show the annotations on the cloned image
        temp_image = annotated_image.copy()
        for contour in annotations:
            points = np.array(contour, dtype=np.int32)
            cv2.polylines(temp_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        for bbox in bbox_annotations:
            x, y, w, h = bbox
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the image with annotations
        cv2.imshow("Image Segmentation", temp_image)

        # Press 's' to save annotations, 'c' to clear, and 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save annotations
            with open("annotations.txt", "a") as f:
                f.write(f"Image: {image_path}\n")
                for bbox in bbox_annotations:
                    f.write(f"x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}\n")
            print("Annotations saved to annotations.txt")
        elif key == ord("c"):
            # Clear annotations
            annotations.clear()
            bbox_annotations.clear()
            annotated_image = image.copy()
            print("Annotations cleared")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

# Main function to process multiple images
if __name__ == "__main__":
    PathNames = r"C:\Users\cic\Desktop\Local_Repo\Image_dataset"
    FileNames = [f for f in os.listdir(PathNames) if f.endswith(('jpg', 'png'))]

    for file_name in FileNames:
        image_path = os.path.join(PathNames, file_name)
        print(f"Processing {file_name}")
        segment_image(image_path)