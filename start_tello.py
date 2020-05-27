from tello import Tello
from TelloVideo import TelloVideo
import time
    
def main():
    tello = Tello()
    if tello.connect():
        tello_video = TelloVideo(tello)
        tello_video.start()
        while not tello_video.has_stopped:
            time.sleep(0.5)
            
if __name__ == "__main__":
    main()