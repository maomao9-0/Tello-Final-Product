from PIL import Image
import numpy as np
import cv2
import threading
import sys
import time


class TelloVideo:
    POSE_THRESHOLD = 3
    POSE_TIME = 2
    POSE_CONFIDENCE = 0.8
    YOLO_CONFIDENCE = 0.6
    YOLO_THRESHOLD = 0.3
    MIN_HUMAN_WIDTH = 3

    RED = [0, 0, 255]
    BLUE = [255, 0, 0]

    def __init__(self, tello):
        self.tello = tello
        self.video_thread = threading.Thread(target=self._video_loop, args=())
        self.predict_thread = threading.Thread(target=self._predict_loop, args=())
        self.tello_thread = threading.Thread(target=self._tello_loop, args=())
        self.stop_turning = False
        self.has_stopped = False

        # YOLO
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        # derive the paths to the YOLO weights and model configuration
        weights_path = "yolo-coco/yolov3.weights"
        config_path = "yolo-coco/yolov3.cfg"
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def start(self):
        while not self.tello.in_command_mode:
            time.sleep(0.5)
        self.video_thread.start()
        self.tello_thread.start()
        self.predict_thread.start()
        while self.tello.in_command_mode:
            time.sleep(0.5)
        self.has_stopped = True
        self._close()

    def _video_loop(self):
        self.tello.video_mode_on()
        for i in range(20):
            if self.tello.in_video_mode:
                break
            time.sleep(0.5)
        if not self.tello.in_video_mode:
            print("Unable to connect to drone camera!")
            return

        cv2.startWindowThread()
        cv2.namedWindow("Tello Live View")

        while not self.has_stopped:
            frame = self.tello.readframe()
            if frame is None or frame.size == 0:
                continue
            image = Image.fromarray(frame)
            converted_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow("Tello Live View", converted_image)
            cv2.waitKey(1)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.YOLO_CONFIDENCE, self.YOLO_THRESHOLD)

            self.stop_turning = False

            if len(idxs) > 0:
                idxs = idxs.flatten()
                print(idxs)

                is_too_close = [False] * len(idxs)
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        idx1 = idxs[i]
                        (x1, y1) = (self.boxes[idx1][0], self.boxes[idx1][1])
                        (w1, h1) = (self.boxes[idx1][2], self.boxes[idx1][3])
                        idx2 = idxs[j]
                        (x2, y2) = (self.boxes[idx2][0], self.boxes[idx2][1])
                        (w2, h2) = (self.boxes[idx2][2], self.boxes[idx2][3])
                        ave_w = (w1 + w2) / 2
                        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                        if dist < ave_w * self.MIN_HUMAN_WIDTH:
                            is_too_close[i] = True
                            is_too_close[j] = True

                # loop over the indexes we are keeping
                for i in range(len(idxs)):
                    idx = idxs[i]
                    # extract the bounding box coordinates
                    (x, y) = (self.boxes[idx][0], self.boxes[idx][1])
                    (w, h) = (self.boxes[idx][2], self.boxes[idx][3])
                    # draw a bounding box rectangle and label on the frame
                    color = self.RED if is_too_close[i] else self.BLUE

                    cv2.rectangle(converted_image, (x, y), (x + w, y + h), color, 2)
                    if is_too_close[i]:
                        text = "TOO CLOSE!!"
                        cv2.putText(converted_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        self.stop_turning = True

            cv2.imshow("Tello Live Camera", converted_image)  # do not remove this line. Without this line that is no video.

            # check if any key is pressed and what is the key. If no key pressed within 1 sec, it will return -1
            k = cv2.waitKey(1)

            if 'q' == chr(k & 255):
                self._close()

    def _tello_loop(self):
        # self.tello.instruct("takeoff")
        # self.tello.instruct("battery?")
        while not self.has_stopped:
            if self.stop_turning:
                # self.tello.instruct("cw 30")
                print("Stop turning")
                time.sleep(1)
        # self.tello.instruct("land")

    def _predict_loop(self):
        self.tello.video_mode_on()
        for i in range(20):
            if self.tello.in_video_mode:
                break
            time.sleep(0.5)
        if not self.tello.in_video_mode:
            print("Unable to connect to drone camera!")
            return

        while not self.has_stopped and self.tello.in_video_mode:
            time.sleep(1)
            frame = self.tello.readframe()
            if frame is None or frame.size == 0:
                continue
            image = Image.fromarray(frame)

            # YOLO
            converted_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            (H, W, C) = converted_image.shape
            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv2.dnn.blobFromImage(converted_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.layer_names)
            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            self.boxes = []
            self.confidences = []

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
                    if class_id == 0 and confidence > self.YOLO_CONFIDENCE:
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
                        self.boxes.append([x, y, int(width), int(height)])
                        self.confidences.append(float(confidence))

    def _close(self):
        self.has_stopped = True
        self.tello.instruct("land")
        cv2.destroyAllWindows()
        self.tello.video_freeze()
        self.tello.close()
        sys.exit()
