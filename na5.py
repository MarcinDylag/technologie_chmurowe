from flask import Flask, flash, redirect, url_for, render_template, request, session
import cv2
import urllib.request
import base64

# Initializing the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initializing the app object and defining secret key for sessions
app = Flask(__name__)
app.secret_key = 'e2129d86c546884ac0b94218b6b41b9a18b68a1ad4d6ad2b2136de20a26724da'


# source page:
# https://gist.github.com/patharanordev/e22f2fe1c2593c9e5d9f7fbd00d8da09
# Encodes an image as a png and encodes to base 64 for display
def encode_img(img, im_type):
    success, encoded_img = cv2.imencode('.{}'.format(im_type), img)
    if success:
        return base64.b64encode(encoded_img).decode()
    return ''


# Most of below code - source page:
# https://www.youtube.com/watch?v=9MHYHgh4jYc&list=PLzMcBGfZo4-n4vJJybUVV3Un_NFS5EOgX&index=4
# Display home page
@app.route("/", methods=['GET'])
def home():
    img = 'static/station.jpg'
    session['image'] = img
    if request.method == "GET":
        return render_template("index.html", image=img)


# Run recognition API
@app.route("/recognize")
def recognize():
    if request.method == "GET":
        if 'image' in session:
            img = session['image']
            if 'static/' in img:
                reco_img = cv2.imread(img)
            else:
                file_name = 'download.jpg'
                urllib.request.urlretrieve(img, file_name)
                reco_img = cv2.imread(file_name)
            # resize image and recognize people
            reco_img = cv2.resize(reco_img, (800, 500))
            (rects, weights) = hog.detectMultiScale(reco_img, winStride=(3, 3), padding=(4, 4), scale=1.05)
            peopleCount = len(rects)
            # draw the bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(reco_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # encode image to display it on page
            encoded_img = encode_img(reco_img, 'jpg')
            b64_src = 'data:image/jpeg;base64,'
            img_src = b64_src + encoded_img

            return render_template("recognize.html", image=img_src, people=peopleCount)
        else:
            return redirect(url_for("home"))


# Upload new image with provided URL
@app.route("/change", methods=["POST", "GET"])
def change():
    if request.method == "POST":
        img = request.form["www"]
        # check if form is empty
        if not request.form.get("www"):
            flash("Please provide URL address!", "info")
            return redirect('change')
        else:
            session['image'] = img
            return redirect(url_for("newimage", url=img))
    else:
        return render_template("change.html")


# Display uploaded image
@app.route("/newimage/<path:url>")
def newimage(url):
    return render_template("index.html", image=url)


# Run application
if __name__ == '__main__':
    app.run(debug=True)
