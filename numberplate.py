import easyocr
import cv2 as cv
import imutils
import numpy as np

# reading image and conversion to grayscale:
img = cv.imread('car_3.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


#edge detection:
bfilter = cv.bilateralFilter(gray,11,17,17)
edge = cv.Canny(bfilter,30,200) # applied canny edge detector

# contour detection
# Find the points of contour and stored them in form of tree
keypoints = cv.findContours(edge.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(keypoints)
# sorted them on basis of area and grabed top 10
contours = sorted(contours,key= cv.contourArea, reverse = True)[:10]

location = None
for contour in contours:
    approx= cv.approxPolyDP(contour,10,True)
    #  if our approx has 4 keys points it is our numberplate
    if len(approx) == 4:
        location = approx
        break
print(location)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv.drawContours(mask,[location],0,255,-1)
new_image = cv.bitwise_and(img, img, mask = mask)

(x,y) = np.where(mask ==255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]


reader = easyocr.Reader(['en'])
result  =reader.readtext(cropped_image)
print(result)

text = result[0][-2]
font = cv.FONT_HERSHEY_SIMPLEX
res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60),
                  fontFace=font, fontScale=1, color=(0,255,0),
                  thickness=2, lineType=cv.LINE_AA)
res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)


cv.imshow("gray",res)
cv.waitKey(0)