import cv2 as cv
import numpy as np
import face_recognition

imgRock = face_recognition.load_image_file('Images/bill gates.jpg')
imgRock = cv.cvtColor(imgRock, cv.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/dwayne johnson test.jpg')
imgTest = cv.cvtColor(imgTest, cv.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRock)[0]
encodeFace = face_recognition.face_encodings(imgRock)[0]
# print(faceLoc)
cv.rectangle(imgRock, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeFaceTest = face_recognition.face_encodings(imgTest)[0]
# print(faceLocTest)
cv.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeFace], encodeFaceTest)
faceDistance = face_recognition.face_distance([encodeFace], encodeFaceTest)
print(result, faceDistance)

cv.putText(imgTest, f'{result} {round(faceDistance[0],2)}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv.imshow('the rock', imgRock)
cv.imshow('the Rock test', imgTest)
cv.waitKey(0)
