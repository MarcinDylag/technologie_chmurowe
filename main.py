import cv2
import imutils

# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


img = cv2.imread('peron2.jpg')
img = cv2.resize(img, (800,500))

#print(type(img))
#print(img.shape)

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Resizing the Image
#image = imutils.resize(image,width=min(400, image.shape[1]))

# Detecting all the regions in the
# Image that has a pedestrians inside it
(rects, weights) = hog.detectMultiScale(img,
                                    winStride=(4, 4),
                                    padding=(4, 4),
                                    scale=1.05)

# Drawing the regions in the Image
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y),
                  (x + w, y + h),
                  (0, 0, 255), 2)

# Showing the output Image

print(f'Found {len(rects)} humans')
cv2.imshow("Image", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
