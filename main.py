import cv2
# API
from flask import Flask
from flask_restful import Resource, Api
# Initializing the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# API
app = Flask(__name__)
api = Api(app)

url = 'https://vs-static.virtualspeech.com/img/env/lecture-hall-full.jpg'
class PeopleCounter(Resource):
    def get(self):
        # Loading and Resizing the Image
        img = cv2.imread('peron2.jpg')
        img = cv2.resize(img, (800, 500))
        # Detecting all the regions in the image with people in it
        (rects, weights) = hog.detectMultiScale(img, winStride=(3, 3), padding=(4, 4), scale=1.05)

        return {'people count': len(rects)}


api.add_resource(PeopleCounter, '/')

if __name__ == '__main__':
    app.run(debug=True)

# for (x, y, w, h) in rects:
# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Showing the output Image
# print(f'Found {len(rects)} humans')
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
