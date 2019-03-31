import cv2
import numpy as np
import time
from skimage.feature import hog
from sklearn.externals import joblib
from nms import nms
import argparse

def appendRects(i, j, conf, c, rects):
    x = int((j)*pow(scaleFactor, c))
    y = int((i)*pow(scaleFactor, c))
    w = int((64)*pow(scaleFactor, c))
    h = int((128)*pow(scaleFactor, c))
    rects.append((x, y, conf, w, h))

parser = argparse.ArgumentParser(description='To read image name')

parser.add_argument('-i', "--image", help="Path to the test image", required=True)
parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.5, type=float)
parser.add_argument('-v', '--visualize', help="Visualize the sliding window", action="store_true")
parser.add_argument('-w', '--winstride', help="Pixels to move in one step, in any direction", default=8, type=int)
parser.add_argument('-n', '--nms_threshold', help="Threshold Values between 0 to 1 for NMS thresholding. Default is 0.2", default=0.2, type=float)
args = vars(parser.parse_args())



clf = joblib.load("person_final.pkl")


orig = cv2.imread(args["image"])

img = orig.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


scaleFactor = args["downscale"]
inverse = 1.0/scaleFactor
winStride = (args["winstride"], args["winstride"])
winSize = (128, 64)

rects = []

h, w = gray.shape
count = 0
while (h >= 128 and w >= 64):

    print gray.shape

    h, w= gray.shape
    horiz = w - 64
    vert = h - 128
    print (horiz, vert)
    i = 0
    j = 0
    while i < vert:
        j = 0
        while j < horiz:

            portion = gray[i:i+winSize[0], j:j+winSize[1]]
            features = hog(portion, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2")

            result = clf.predict([features])

            if args["visualize"]:
                visual = gray.copy()
                cv2.rectangle(visual, (j, i), (j+winSize[1], i+winSize[0]), (0, 0, 255), 2)
                cv2.imshow("visual", visual)
                cv2.waitKey(1)

            if int(result[0]) == 1:
                print (result, i, j)
                confidence = clf.decision_function([features])
                appendRects(i, j, confidence, count, rects)


            j = j + winStride[0]

        i = i + winStride[1]

    gray = cv2.resize(gray, (int(w*inverse), int(h*inverse)), interpolation=cv2.INTER_AREA)
    count = count + 1
    print count

print rects

nms_rects = nms(rects, args["nms_threshold"])

for (a, b, conf, c, d) in rects:
    cv2.rectangle(orig, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imshow("Before NMS", orig)
cv2.waitKey(0)



for (a, b, conf, c, d) in nms_rects:
    cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imshow("After NMS", img)
cv2.waitKey(0)
