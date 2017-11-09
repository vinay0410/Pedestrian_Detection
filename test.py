import cv2
from sklearn import svm
import os
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog

clf = joblib.load('person_hard.pkl')
pos_img_dir = "test/pos/"
neg_img_dir = "test/neg/"


def read_filenames():

    f_pos = []
    f_neg = []

    mypath_pos = pos_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_pos):
        f_pos.extend(filenames)
        break
    mypath_neg = neg_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_neg):
        f_neg.extend(filenames)
        break

    print "Positive Image Samples: " + str(len(f_pos))
    print "Negative Image Samples: " + str(len(f_neg))

    return f_pos, f_neg

def read_images(f_pos, f_neg):

    array_pos_features = []
    array_neg_features = []

    for imgfile in f_pos:
        img = cv2.imread(mypath_pos+imgfile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2")
        array_pos_features.append(features)
        total = total + 1

    for imgfile in f_neg:
        img = cv2.imread(mypath_neg+imgfile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2")
        array_neg_features.append(features)
        total = total + 1

    return array_pos_features, array_neg_features



pos_img_files, neg_img_files = read_filenames()

pos_features, neg_features = read_images(pos_img_files, neg_img_files)

true_positives = cv2.countNonZero(clf.predict(pos_features))

false_positives = cv2.countNonZero(clf.predict(neg_features))

print ("True Negatives: " + str(total - true_positives), "False_positives: " + str(false_positives))
