import cv2
import sys
import os.path
from glob import glob
import datetime
def detect(cascade_file="../lbpcascade_animeface.xml"):
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    input_movie = cv2.VideoCapture("video/honglajiao.mp4")
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    frame_number=1
    starttime = datetime.datetime.now()
	
    while True:
    # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1
    # Quit when the input video file ends
        if not ret:
            break
        cascade = cv2.CascadeClassifier(cascade_file)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48))
        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y: y + h, x:x + w, :]
            face = cv2.resize(face, (256, 256))
            save_filename = '%s-%d-%d.png' % ('honglajiao',frame_number,i )
            cv2.imwrite("faces/" + save_filename, face)
    # All done!
    endtime = datetime.datetime.now()
    print(endtime)
    print((endtime - starttime).seconds)
    input_movie.release()


if __name__ == '__main__':
    detect()
    
	
    
    



