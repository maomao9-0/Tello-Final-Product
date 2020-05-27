from PIL import Image
import numpy as np
import cv2
import threading
import sys
import time
import tensorflow


class TelloVideo:
    ERROR_THRESHOLD = 3
    HOLD_TIME = 2

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
            if pred_result == 1:
                # raising hands
                print("Someone is raising hands. Go help him!")
            elif pred_result == 2:
                # neutral person
                print("Someone does not need my help :(")
            elif pred_result == 0:
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
        while self.stopApp == False and self.tello.in_video_mode:
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

            if prediction[pred_result] > 0.8:
                if prev_pred_result == -1:
                    prev_pred_result = pred_result
                    prev_pred_time = time.time()
                elif prev_pred_result != pred_result:
                    error += 1
                    if (error > self.ERROR_THRESHOLD):
                        prev_pred_result = -1
                        prev_pred_time = time.time()
                else:
                    error = 0
                    if time.time() - prev_pred_time > self.HOLD_TIME:
                        self.pred_result = pred_result
                        prev_pred_result = -1
                        prev_pred_time = time.time()
            else:
                error += 1
                if (error > self.ERROR_THRESHOLD):
                    prev_pred_result = -1
                    prev_pred_time = time.time()

    def _close(self):
        self.has_stopped = True
        self.tello.instruct("land")
        cv2.destroyAllWindows()
        self.tello.video_freeze()
        self.tello.close()
        sys.exit()
