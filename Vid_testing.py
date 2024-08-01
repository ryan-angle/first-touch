import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# YOLOv8
model = YOLO('yolov8s.pt')

# List to store bounding boxes
bboxes = []
player_bbox, ball_bbox = None, None
player_tracker, ball_tracker = None, None
tracking_player, tracking_ball = False, False
touch = False
frame = None
lower_range = (np.array([0, 100, 100], np.uint8), np.array([10, 255, 255], np.uint8))
upper_range = (np.array([160, 100, 100], np.uint8), np.array([180, 255, 255], np.uint8))


def detect_players(image, conf_threshold=0.5):
    results = model(image)
    bboxes = []
    # bboxes = results.xyxy[0].cpu().numpy()  # Extract bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        confs = result.boxes.conf.cpu().numpy()  # Extract confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Extract class IDs
        for box, conf, cls in zip(boxes, confs, classes):
            if conf > conf_threshold:  # Filter boxes by confidence score
                xmin, ymin, xmax, ymax = box
                bboxes.append((xmin, ymin, xmax, ymax, conf, cls))
    return bboxes


def draw_boxes(image, bboxes):
    filtered_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, conf, cls = bbox
        label = int(cls)
        if label == 32:
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        else:
            print(cls)
            player_region = image[int(ymin):int(ymax), int(xmin):int(xmax)]

            # Check if the player is wearing the specified color
            if is_color_in_range(player_region, lower_range, upper_range):
                filtered_bboxes.append((xmin, ymin, xmax, ymax))

            for (xmin, ymin, xmax, ymax) in filtered_bboxes:
                # Draw a rectangle around the player
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    return image


# Define function to check if a color is in the specified range
def is_color_in_range(region, low, high):
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_region, low[0], low[1])
    mask2 = cv2.inRange(hsv_region, high[0], high[1])
    mask = mask1 | mask2
    return np.sum(mask) > 0


def on_mouse_click(event, x, y, flags, param):
    global player_tracker, tracking_player, frame, player_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, conf, cls = bbox
            if xmin < x < xmax and ymin < y < ymax:
                print(f"Player clicked at ({x}, {y})")
                player_tracker = cv2.legacy.TrackerKCF_create()
                player_tracker.init(frame, (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)))
                player_bbox = (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))
                tracking_player = True
                break


def main():
    global bboxes, player_tracker, tracking_player, frame, ball_tracker, tracking_ball, player_bbox, ball_bbox, touch
    input_vid = 'test_vids/ryan_mlsnext.mov'
    cap = cv2.VideoCapture(input_vid)

    cv2.namedWindow('Player Detection')
    cv2.setMouseCallback('Player Detection', on_mouse_click)

    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"input fps: {input_fps}")

    scaling_factor_vid = 640 / vid_width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_vids/ryan_example.mp4', fourcc, input_fps, (640, int(vid_height * scaling_factor_vid)))

    freeze_frame = deque(maxlen=150)
    tracking_flags = deque(maxlen=150)
    bounding_boxes = deque(maxlen=150)

    all_frames = []

    # stop tracking once player identified so process is faster?

    # try YOLOv8 tracking? - https://docs.ultralytics.com/modes/track/#persisting-tracks-loop
    while cap.isOpened():
        ret, original_frame = cap.read()
        try:
            height, width = original_frame.shape[:2]
        except AttributeError:
            # save video to filepath
            break

        # Calculate the scaling factor
        scaling_factor = 640 / width

        # Resize the image
        new_dimensions = (640, int(height * scaling_factor))
        original_frame = cv2.resize(original_frame, new_dimensions, interpolation=cv2.INTER_AREA)
        frame = original_frame.copy()
        freeze_frame.append(original_frame)
        if not ret:
            break

        if not tracking_player:
            bboxes = detect_players(frame)
            frame = draw_boxes(frame, bboxes)
            tracking_flags.append(False)
            bounding_boxes.append(None)
        else:
            success_player, box_player = player_tracker.update(frame)
            if success_player:
                player_bbox = (int(box_player[0]), int(box_player[1]), int(box_player[2]), int(box_player[3]))
                xmin, ymin, w, h = [int(v) for v in box_player]
                cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 2)
                tracking_flags.append(True)
                bounding_boxes.append(player_bbox)
            else:
                tracking_player = False  # Stop tracking if the tracker fails
                tracking_flags.append(False)
                bounding_boxes.append(None)

            if not tracking_ball:
                bboxes = detect_players(frame)
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax, conf, cls = bbox
                    if int(cls) == 32:  # YOLO class ID for sports ball
                        ball_tracker = cv2.legacy.TrackerKCF_create()
                        ball_tracker.init(frame, (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)))
                        ball_bbox = (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))
                        tracking_ball = True
                        break

            if tracking_ball:
                success_ball, box_ball = ball_tracker.update(frame)
                if success_ball:
                    ball_bbox = (int(box_ball[0]), int(box_ball[1]), int(box_ball[2]), int(box_ball[3]))
                    cv2.rectangle(frame, (ball_bbox[0], ball_bbox[1]),
                                  (ball_bbox[0] + ball_bbox[2], ball_bbox[1] + ball_bbox[3]), (255, 0, 0), 2)

                    # Check if the ball is inside the player's bounding box
                    # Let's eventually change this to either oldest or player's first on screen appearance or just 0.
                    if not touch and (player_bbox[0] - 10 < ball_bbox[0] < player_bbox[0] + player_bbox[2] + 10 and
                                      player_bbox[1] - 10 < ball_bbox[1] < player_bbox[1] + player_bbox[3] + 10):
                        print("Ball has entered the player's tracking frame!")
                        touch = True
                        oldest_frame = freeze_frame.popleft()
                        oldest_tracking = tracking_flags.popleft()
                        oldest_bbox = bounding_boxes.popleft()
                        to_insert = -1
                        if not oldest_tracking:
                            while not oldest_tracking:
                                oldest_tracking = tracking_flags.popleft()
                                oldest_frame = freeze_frame.popleft()
                                oldest_bbox = bounding_boxes.popleft()
                        for index, cur_frame in enumerate(all_frames):
                            if np.array_equal(cur_frame, oldest_frame):
                                to_insert = index
                                break
                        print(to_insert)
                        # print(oldest_bbox)
                        center_x = oldest_bbox[0] + oldest_bbox[2] // 2
                        center_y = oldest_bbox[1] + oldest_bbox[3] // 2

                        print(center_x, center_y)

                        radius = max(oldest_bbox[2], oldest_bbox[3]) // 2
                        cv2.circle(oldest_frame, (center_x, center_y), radius + 20, (0, 0, 255), 2)

                        # basically just add the circle to the player only in oldest_frame
                        all_frames = all_frames[:to_insert + 1] + [oldest_frame] * 100 + all_frames[to_insert + 1:]
                        # Add extra frames for the freeze, add 'not touch' to if statement
                        # Create queue, add up to (20) frames, add current frame and remove the top if full
                        # Upon first time ball is touched, pop the oldest frame from the queue.
                else:
                    tracking_ball = False  # Stop tracking if the tracker fails

        cv2.imshow('Player Detection', frame)
        all_frames.append(original_frame)
        # out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for f in all_frames:
        out.write(f)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
