import time
import numpy as np
import cv2
import sys
from tello import Tello
from PIL import Image

POSE_THRESHOLD = 3
POSE_TIME = 2
POSE_CONFIDENCE = 0.8
YOLO_CONFIDENCE = 0.6
YOLO_THRESHOLD = 0.3
MIN_HUMAN_WIDTH = 4

RED = [0, 0, 255]
BLUE = [255, 0, 0]


def main():
    np.random.seed(42)
    # derive the paths to the YOLO weights and model configuration
    weights_path = "yolo-coco/yolov3.weights"
    config_path = "yolo-coco/yolov3.cfg"
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cam = cv2.VideoCapture(0)

    '''
    while not tello.in_command_mode:
        time.sleep(0.5)
    tello.video_mode_on()

    for i in range(20):
        if tello.in_video_mode:
            break
        time.sleep(0.5)
    if not tello.in_video_mode:
        print("Unable to connect to drone camera!")
        return
    '''

    while True:
        #'''
        ret, converted_image = cam.read()
        if not ret:
            continue
        #'''

        '''
        # tello.instruct("battery?")
        frame = tello.readframe()
        if frame is None or frame.size == 0:
            continue
        image = Image.fromarray(frame)
        converted_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        '''

        (H, W, C) = converted_image.shape
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(converted_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(layer_names)
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                # make sure that class_id is 0 to identify only people
                if class_id == 0 and confidence > YOLO_CONFIDENCE:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONFIDENCE, YOLO_THRESHOLD)

            if len(idxs) > 0:
                idxs = idxs.flatten()
                is_too_close = [False] * len(idxs)
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        print("%s %s" % (i, j))
                        idx1 = idxs[i]
                        (x1, y1) = (boxes[idx1][0], boxes[idx1][1])
                        (w1, h1) = (boxes[idx1][2], boxes[idx1][3])
                        idx2 = idxs[j]
                        (x2, y2) = (boxes[idx2][0], boxes[idx2][1])
                        (w2, h2) = (boxes[idx2][2], boxes[idx2][3])
                        ave_w = (w1 + w2) / 2
                        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                        if dist < ave_w * MIN_HUMAN_WIDTH:
                            is_too_close[i] = True
                            is_too_close[j] = True

                # loop over the indexes we are keeping
                for i in range(len(idxs)):
                    idx = idxs[i]
                    # extract the bounding box coordinates
                    (x, y) = (boxes[idx][0], boxes[idx][1])
                    (w, h) = (boxes[idx][2], boxes[idx][3])
                    # draw a bounding box rectangle and label on the frame
                    color = RED if is_too_close[i] else BLUE

                    cv2.rectangle(converted_image, (x, y), (x + w, y + h), color, 2)
                    if is_too_close[i]:
                        text = "TOO CLOSE!!"
                        cv2.putText(converted_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("TelloLiveview", converted_image)  # do not remove this line. Without this line that is no video.

        # check if any key is pressed and what is the key. If no key pressed within 1 sec, it will return -1
        k = cv2.waitKey(1)

        if 'q' == chr(k & 255):
            cv2.destroyAllWindows()
            sys.exit()


'''
tello = Tello()
if tello.connect():
    main()
'''

main()