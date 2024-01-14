import cv2
import urllib.request
# API
from flask import Flask, request
from flask_restful import Resource, Api
# Initializing the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# API
app = Flask(__name__)
api = Api(app)

name = 'download.jpg'


class PeopleCounterStatic(Resource):
    def get(self):
        # Loading and Resizing the Image
        img = cv2.imread('peron2.jpg')
        img = cv2.resize(img, (800, 500))
        # Detecting all the regions in the image with people in it
        (rects, weights) = hog.detectMultiScale(img, winStride=(3, 3), padding=(4, 4), scale=1.05)

        return {'peopleCount': len(rects)}


class PeopleCounterDynamic(Resource):

    def get(self):
        url = request.args.get('url')
        urllib.request.urlretrieve(url, name)
        img2 = cv2.imread(name)
        img2 = cv2.resize(img2, (800, 500))
        (rects, weights) = hog.detectMultiScale(img2, winStride=(3, 3), padding=(4, 4), scale=1.05)
        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamic, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)
