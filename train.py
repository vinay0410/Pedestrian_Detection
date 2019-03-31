import cv2
from sklearn import svm
import os
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.utils import shuffle
import sys
import argparse
import random

MAX_HARD_NEGATIVES = 20000

parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--pos', help='Path to directory containing Positive Images')
parser.add_argument('--neg', help='Path to directory containing Negative images')

args = parser.parse_args()
pos_img_dir = args.pos
neg_img_dir = args.neg


def crop_centre(img):
    h, w, _ = img.shape
    l = (w - 64)/2
    t = (h - 128)/2

    crop = img[t:t+128, l:l+64]
    return crop

def ten_random_windows(img):
    h, w = img.shape
    if h < 128 or w < 64:
        return []

    h = h - 128;
    w = w - 64

    windows = []

    for i in range(0, 10):
        x = random.randint(0, w)
        y = random.randint(0, h)
        windows.append(img[y:y+128, x:x+64])

    return windows


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

    return f_pos, f_neg


def read_images(pos_files, neg_files):

    X = []
    Y = []

    pos_count = 0

    for img_file in pos_files:
        print os.path.join(pos_img_dir, img_file)
        img = cv2.imread(os.path.join(pos_img_dir, img_file))

        cropped = crop_centre(img)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        pos_count += 1

        X.append(features)
        Y.append(1)


    neg_count = 0

    for img_file in neg_files:
        print os.path.join(neg_img_dir, img_file)
        img = cv2.imread(os.path.join(neg_img_dir, img_file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        windows = ten_random_windows(gray_img)

        for win in windows:
            features = hog(win, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            neg_count += 1
            X.append(features)
            Y.append(0)


    return X, Y, pos_count, neg_count


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0]-128, step_size[1]):
        for x in xrange(0, image.shape[1]-64, step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def hard_negative_mine(f_neg, winSize, winStride):

    hard_negatives = []
    hard_negative_labels = []

    count = 0
    num = 0
    for imgfile in f_neg:

        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, im_window) in sliding_window(gray, winSize, winStride):
            features = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            if (clf1.predict([features]) == 1):
                hard_negatives.append(features)
                hard_negative_labels.append(0)

                count = count + 1

            if (count == MAX_HARD_NEGATIVES):
                return np.array(hard_negatives), np.array(hard_negative_labels)

        num = num + 1

        sys.stdout.write("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(round((count / float(MAX_HARD_NEGATIVES))*100, 4)) + " %" )

        sys.stdout.flush()

    return np.array(hard_negatives), np.array(hard_negative_labels)




pos_img_files, neg_img_files = read_filenames()

print "Total Positive Images : " + str(len(pos_img_files))
print "Total Negative Images : " + str(len(neg_img_files))
print "Reading Images"

X, Y, pos_count, neg_count = read_images(pos_img_files, neg_img_files)

X = np.array(X)
Y = np.array(Y)

X, Y = shuffle(X, Y, random_state=0)


print "Images Read and Shuffled"
print "Positives: " + str(pos_count)
print "Negatives: " + str(neg_count)
print "Training Started"

clf1 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)


clf1.fit(X, Y)
print "Trained"


joblib.dump(clf1, 'person_pre-eliminary.pkl')


print "Hard Negative Mining"

winStride = (8, 8)
winSize = (64, 128)

print ("Maximum Hard Negatives to Mine: " + str(MAX_HARD_NEGATIVES))

hard_negatives, hard_negative_labels = hard_negative_mine(neg_img_files, winSize, winStride)

sys.stdout.write("\n")

hard_negatives = np.concatenate((hard_negatives, X), axis = 0)
hard_negative_labels = np.concatenate((hard_negative_labels, Y), axis = 0)

hard_negatives, hard_negative_labels = shuffle(hard_negatives, hard_negative_labels, random_state=0)

print "Final Samples Dims: " + str(hard_negatives.shape)
print "Retraining the classifier with final data"

clf2 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)

clf2.fit(hard_negatives, hard_negative_labels)

print "Trained and Dumping"

joblib.dump(clf2, 'person_final.pkl')
