import cv2
from sklearn import svm
import os
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
import sys
import argparse
import random

MAX_HARD_NEGATIVES = 120725

parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--pos', help='Path to directory containing Positive Images')
parser.add_argument('--neg', help='Path to directory containing Negative images')

args = parser.parse_args()
pos_img_dir = args.pos
neg_img_dir = args.neg


def crop_centre(img):
    h, w, d = img.shape
    l = (w - 64)/2
    t = (h - 128)/2
    #print (h, w, l, t)
    crop = img[t:t+128, l:l+64]
    return crop

def ten_random_windows(img):
    h, w, d = img.shape
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

    for img_file in pos_files:
        print os.path.join(pos_img_dir, img_file)
        img = cv2.imread(os.path.join(pos_img_dir, img_file))

        #filename, file_extension = os.path.splitext(mypath_pos + img_file)
        #filename = os.path.basename(filename)
        cropped = crop_centre(img)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True)
        #joblib.dump(features, "features/pos/" + filename + ".feat")
        X.append(features)
        Y.append(1)



    # Loading Negative images

    for img_file in neg_files:
        print os.path.join(neg_img_dir, img_file)
        img = cv2.imread(os.path.join(neg_img_dir, img_file))
        windows = ten_random_windows(img)
        #filename, file_extension = os.path.splitext(mypath_neg + img_file)
        #filename = os.path.basename(filename)
        for win in windows:
            gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
            features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True)
            #joblib.dump(features, "features/neg/" + str(filename) + ".feat")
            X.append(features)
            Y.append(0)

    return X, Y


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


def hard_negative_mine(f_neg, winSize, winStride, hard_negatives, hard_negative_labels):


    count = 0
    num = 0
    for imgfile in f_neg:
        #filename, file_extension = os.path.splitext(neg_img_dir + imgfile)
        #filename = os.path.basename(filename)
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, im_window) in sliding_window(gray, winSize, winStride):
            features = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True)
            if (clf1.predict([features]) == 1):
                hard_negatives[count] = features
                hard_negative_labels[count] = 0
                #joblib.dump(features, "features/neg_mined/" + str(filename) + str(imgcount) + ".feat")
                count = count + 1

            if (count == MAX_HARD_NEGATIVES):
                return

        num = num + 1
        #print "Images Done: " + str(num)
        print (hard_negatives.nbytes, hard_negative_labels.nbytes)

        sys.stdout.write("\r" + "\tHard negatives: " + str(count) + "\tCompleted: " + str((count / float(MAX_HARD_NEGATIVES))*100) + " %" )

        #print "Hard Negatives: " + str(count)
        #if (num == 10):
    #        break
        sys.stdout.flush()

    #objgraph.show_backrefs()

    return hard_negatives, hard_negative_labels




pos_img_files, neg_img_files = read_filenames()

print "Total Positive Images : " + str(len(pos_img_files))
print "Total Negative Images : " + str(len(neg_img_files))
print "Reading Images"

X, Y = read_images(pos_img_files, neg_img_files)

np.random.shuffle(X)
np.random.shuffle(Y)

X = np.array(X)
Y = np.array(Y)

print "Images Read and Shuffled"
print "Training Started"

clf1 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)


clf1.fit(X, Y)
print "Trained"


joblib.dump(clf1, 'person.pkl')


print "Hard Negative Mining"

winStride = (8, 8)
winSize = (64, 128)

hard_negatives = np.empty(shape=[MAX_HARD_NEGATIVES, 3780])
hard_negative_labels = np.empty(shape=[MAX_HARD_NEGATIVES])

print (hard_negatives.nbytes, hard_negative_labels.nbytes)

hard_negative_mine(neg_img_files, winSize, winStride, hard_negatives, hard_negative_labels)


sys.stdout.write("\n")

print (hard_negatives.shape, hard_negative_labels.shape)

hard_negatives = np.concatenate((hard_negatives, X), axis = 0)
hard_negative_labels = np.concatenate((hard_negative_labels, Y), axis = 0)

print "Final Samples: " + str(hard_negatives.shape)
print "Retraining the classifier with final data"

clf2 = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)

clf2.fit(hard_negatives, hard_negative_labels)

print "Trained and Dumping"

joblib.dump(clf2, 'person_final.pkl')
