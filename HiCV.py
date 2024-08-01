import cv2
import torch
import numpy as np
from sklearn.metrics import pairwise_distances

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x

# List to store bounding boxes
bboxes = []
img = None
base_image = None
selected_player_features = None
tracker, tracking, frame = None, None, None


# Define a function to detect players
def detect_players(image_path, lower_range, upper_range):
    global img, base_image
    global bboxes
    base_image = cv2.imread(image_path)

    height, width = base_image.shape[:2]

    # Calculate the scaling factor
    scaling_factor = 640 / width

    # Resize the image
    new_dimensions = (640, int(height * scaling_factor))
    img = cv2.resize(base_image, new_dimensions, interpolation=cv2.INTER_AREA)
    base_image = cv2.resize(base_image, new_dimensions, interpolation=cv2.INTER_AREA)

    results = model(img)
    bboxes = results.xyxy[0].numpy()  # Extract bounding boxes

    # Render results on the image
    # results.render()

    filtered_bboxes = []
    # Draw bounding boxes on the image
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, conf, cls = bbox

        # Extract player region
        player_region = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        # Check if the player is wearing the specified color
        if is_color_in_range(player_region, lower_range, upper_range):
            filtered_bboxes.append((xmin, ymin, xmax, ymax))

        for (xmin, ymin, xmax, ymax) in filtered_bboxes:
            # Draw a rectangle around the player
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    # Ensure the entire image is displayed
    # height, width = img.shape[:2]
    # resized_img = cv2.resize(img, (width, height))

    cv2.imshow('Player Detection', img)
    cv2.setMouseCallback('Player Detection', on_mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for img in results.ims:
    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imshow('Player Detection', img_bgr)
    #     cv2.setMouseCallback('Player Detection', on_mouse_click)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return results


# Define mouse click event handler
def on_mouse_click(event, x, y, flags, param):
    global base_image, selected_player_features
    if event == cv2.EVENT_LBUTTONDOWN:
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, conf, cls = bbox
            if xmin < x < xmax and ymin < y < ymax:
                print(f"Player clicked at ({x}, {y})")
                # Here you can add code to handle the clicked player
                # For example, highlight the selected player, collect data, etc.

                # Draw a circle around the clicked player
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)
                radius = int(max(xmax - xmin, ymax - ymin) / 2)
                cv2.circle(img, (center_x, center_y), radius+10, (0, 0, 255), 2)
                # Save the modified image
                cv2.imwrite('output_imgs/highlighted_player.png', img)
                # Display the updated image
                cv2.imshow('Player Detection', img)

                # Extract features of the selected player
                player_roi = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                selected_player_features = extract_features(player_roi)

                cv2.waitKey(500)
                # Close all OpenCV windows
                cv2.destroyAllWindows()
                break


# Function to extract features from the player region
def extract_features(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


# Function to find the selected player in other images
def find_player_in_other_images(image_paths, lower_range, upper_range):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        scaling_factor = 640 / width
        new_dimensions = (640, int(height * scaling_factor))
        img_resized = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        results = model(img_resized)
        bboxes = results.xyxy[0].numpy()
        differences = {}
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, conf, cls = bbox
            player_region = img_resized[int(ymin):int(ymax), int(xmin):int(xmax)]
            if is_color_in_range(player_region, lower_range, upper_range):
                print(f"Is in color range")
                player_features = extract_features(player_region)
                if player_features is not None and selected_player_features is not None:
                    distances = pairwise_distances(player_features, selected_player_features, metric='euclidean')
                    mean_distance = np.mean(distances)
                    differences[mean_distance] = (xmin, ymin, xmax, ymax)
                    print(mean_distance)
                    if mean_distance < 0.5:  # Threshold value to determine if the players match
                        cv2.rectangle(img_resized, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                        break
        xmin, ymin, xmax, ymax = differences[min(differences.keys())]
        cv2.rectangle(img_resized, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.imshow('Player Recognition', img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Define function to check if a color is in the specified range
def is_color_in_range(region, low, high):
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_region, low[0], low[1])
    mask2 = cv2.inRange(hsv_region, high[0], high[1])
    mask = mask1 | mask2
    return np.sum(mask) > 0


def main():
    # Define color range for the jerseys (example: blue color)
    # Define color range for the red jerseys
    lower_red_range = (np.array([0, 100, 100], np.uint8), np.array([10, 255, 255], np.uint8))
    upper_red_range = (np.array([160, 100, 100], np.uint8), np.array([180, 255, 255], np.uint8))

    # Test the function
    screenshot_path = 'test_imgs/ryan_test2.PNG'
    results = detect_players(screenshot_path, lower_red_range, upper_red_range)

    other_image_paths = ['test_imgs/ryan_test.PNG']
    find_player_in_other_images(other_image_paths, lower_red_range, upper_red_range)

if __name__ == '__main__':
    main()
