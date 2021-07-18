import cv2
import os
import numpy as np
import faceRecognition as fr
test_img=cv2.imread(r'/Users/chandan/pc/internproject/MS-Dhoni16c0e955558_large.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

faces,faceID=fr.labels_for_training_data('/Users/chandan/pc/internproject/photos')
face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'/Users/chandan/pc/internproject/trainingData.yml')
name={0:'dhoni',1:'chandan'}
for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence",confidence)
    print("label",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):
        continue
    fr.put_text(test_img,predicted_name,x,y)
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
