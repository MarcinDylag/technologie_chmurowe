from flask import Flask, redirect, url_for, render_template, request
from flask_restful import Resource, Api
import cv2
import urllib.request
import base64


def encode_img(img, im_type):
    """Encodes an image as a png and encodes to base 64 for display."""
    success, encoded_img = cv2.imencode('.{}'.format(im_type), img)
    if success:
        return base64.b64encode(encoded_img).decode()
    return ''


# Initializing the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)


# Declare img as a global variable
img = None


@app.route("/", methods=['GET'])
def home():
    global img
    img = 'https://thumbs.dreamstime.com/z/ludzie-na-dworcu-kolejowym-lotnisku-czekaj%C4%85cy-poci%C4%85g-frankfurt-niemcy-maj-259290207.jpg'
    if request.method == "GET":
        return render_template("index.html", image=img)


@app.route("/recognize")
def recognize():
    global img
    if request.method == "GET":
        uploaded = img
        file_name = 'recognized.jpg'
        urllib.request.urlretrieve(uploaded, file_name)
        img2 = cv2.imread(file_name)

        img2 = cv2.resize(img2, (800, 500))
        (rects, weights) = hog.detectMultiScale(img2, winStride=(3, 3), padding=(4, 4), scale=1.05)
        peopleCount = len(rects)

        # draw the bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

        encoded_img = encode_img(img2, 'jpg')
        b64_src = 'data:image/jpeg;base64,'
        img_src = b64_src + encoded_img

        return render_template("recognize.html", image=img_src, people=peopleCount)


@app.route("/change", methods=["POST", "GET"])
def change():
    global img
    if request.method == "POST":
        img = request.form["www"]
        return redirect(url_for("newimage", url=img))
    else:
        return render_template("change.html"), img


@app.route("/newimage/<path:url>")
def newimage(url):
    global img
    img = url
    return render_template("index.html", image=img)


if __name__ == '__main__':
    app.run(debug=True)
