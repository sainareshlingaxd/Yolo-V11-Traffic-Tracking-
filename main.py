import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

# Remove existing output images
files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

from sort import *
tracker = Sort()
memory = {}
counter = 0

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", required=True,
                help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
ap.add_argument("-s", "--speedup_factor", type=int, default=2,
                help="factor by which to speed up the video processing")
args = vars(ap.parse_args())

# Load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()
ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Initialize the video stream
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# Try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# Initialize a set to keep track of counted vehicle IDs
counted_ids = set()

# Loop over frames from the video file stream
frame_counter = 0
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    # Skip frames based on the speedup factor
    if frame_counter % args["speedup_factor"] != 0:
        frame_counter += 1
        continue

    # Capture the original dimensions of the frame
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Prepare the frame for YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

        # Increment the counter for each detected vehicle
        if indexIDs[-1] not in counted_ids:
            counted_ids.add(indexIDs[-1])
            counter += 1

    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            vehicle_type = LABELS[classIDs[i]]

            # Filter to display only vehicles of interest
            if vehicle_type in ['car', 'motorbike', 'bus', 'truck']:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, vehicle_type, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, str(counter), (100, 200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
    cv2.imshow("Frame", frame)

    # Initialize our video writer if not already initialized
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 60, (W, H), True)  # Set to 60 FPS
        print(f"[INFO] Writing video with resolution: {W}x{H}")

    # Write the processed frame to the output video
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1  # Increment the frame counter

print("[INFO] cleaning up...")
if writer is not None:
    writer.release()
vs.release()
cv2.destroyAllWindows()
