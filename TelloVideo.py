from PIL import Image, ImageOps
import numpy as np
import cv2
import threading
import sys
import time
import tensorflow
import os


class TelloVideo:
    POSE_THRESHOLD = 3
    POSE_TIME = 2
    POSE_CONFIDENCE = 0.8
    YOLO_CONFIDENCE = 0.6
    YOLO_THRESHOLD = 0.3
    MIN_HUMAN_WIDTH = 4

    BLUE = [0, 0, 255]
    RED = [255, 0, 0]

    def __init__(self, tello):
        self.tello = tello
        self.video_thread = threading.Thread(target=self._video_loop, args=())
        self.predict_thread = threading.Thread(target=self._predict_loop, args=())
        self.tello_thread = threading.Thread(target=self._tello_loop, args=())
        self.pred_result = -1
        self.has_stopped = False

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # Load the model
        self.model = tensorflow.keras.models.load_model('keras_model.h5')
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # YOLO
        labels_path = "yolo-coco/coco.names"
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

        while not self.has_stopped:
            frame = self.tello.readframe()
            if frame is None or frame.size == 0:
                continue
            image = Image.fromarray(frame)
            cvt_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow("Tello Live View", cvt_image)
            cv2.waitKey(1)

    def _tello_loop(self):
        self.tello.instruct("takeoff")
        while not self.has_stopped:
            if self.pred_result == 1:
                # raising hands
                print("Someone is raising hands. Go help him!")
            elif self.pred_result == 2:
                # neutral person
                print("Someone does not need my help :(")
            elif self.pred_result == 0:
                # background
                print("Nice background :)")
            self.tello.instruct("cw 30")
            time.sleep(1)
        self.tello.instruct("land")

    def _pred_loop(self):
        for i in range(20):
            if self.tello.in_video_mode:
                break
            time.sleep(0.5)
        if not self.tello.in_video_mode:
            print("Unable to connect to drone camera!")
            return
        prev_pred_time = time.time()
        prev_pred_result = -1
        error = 0
        size = (224, 224)
        while not self.stopApp and self.tello.in_video_mode:
            frame = self.tello.readframe()
            if frame is None or frame.size == 0:
                continue
            image = Image.fromarray(frame)
            resized_image = ImageOps.fit(image, size, Image.ANTIALIAS)

            image_array = np.asarray(resized_image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            self.data[0] = normalized_image_array
            # run the inference
            prediction = self.model.predict(self.data)
            # print(prediction)
            # prediction is a numpy array of confidence level of the 3 classes [[0.1,0.2,0.7]]

            pred_result = np.argmax(prediction)

            if prediction[pred_result] > self.POSE_CONFIDENCE:
                if prev_pred_result == -1:
                    prev_pred_result = pred_result
                    prev_pred_time = time.time()
                elif prev_pred_result != pred_result:
                    error += 1
                    if error > self.POSE_THRESHOLD:
                        prev_pred_result = -1
                        prev_pred_time = time.time()
                else:
                    error = 0
                    if time.time() - prev_pred_time > self.POSE_TIME:
                        self.pred_result = pred_result
                        prev_pred_result = -1
                        prev_pred_time = time.time()
            else:
                error += 1
                if error > self.POSE_THRESHOLD:
                    prev_pred_result = -1
                    prev_pred_time = time.time()

            # YOLO
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            (H, W, C) = converted_image.shape
            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv2.dnn.blobFromImage(converted_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.layer_names)
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
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.YOLO_CONFIDENCE, self.YOLO_THRESHOLD).flatten()

                is_too_close = [False] * len(idxs)
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        idx1 = idxs[i]
                        (x1, y1) = (boxes[idx1][0], boxes[idx1][1])
                        (w1, h1) = (boxes[idx1][2], boxes[idx1][3])
                        idx2 = idxs[j]
                        (x2, y2) = (boxes[idx2][0], boxes[idx2][1])
                        (w2, h2) = (boxes[idx2][2], boxes[idx2][3])
                        ave_w = (w1 + w2) / 2
                        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                        if dist < ave_w * self.MIN_HUMAN_WIDTH:
                            is_too_close[i] = True
                            is_too_close[j] = True

                # loop over the indexes we are keeping
                for i in range(len(idxs)):
                    idx = idxs[i]
                    # extract the bounding box coordinates
                    (x, y) = (boxes[idx][0], boxes[idx][1])
                    (w, h) = (boxes[idx][2], boxes[idx][3])
                    # draw a bounding box rectangle and label on the frame
                    color = self.RED if is_too_close[i] else self.BLUE

                    cv2.rectangle(converted_image, (x, y), (x + w, y + h), color, 2)
                    if is_too_close[i]:
                        text = "TOO CLOSE!!"
                        cv2.putText(converted_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("TelloLiveview", converted_image)  # do not remove this line. Without this line that is no video.

            # check if any key is pressed and what is the key. If no key pressed within 1 sec, it will return -1
            k = cv2.waitKey(1)

            if 'q' == chr(k & 255):
                self._close()

    def _close(self):
        self.has_stopped = True
        self.tello.instruct("land")
        cv2.destroyAllWindows()
        self.tello.video_freeze()
        self.tello.close()
        sys.exit()
