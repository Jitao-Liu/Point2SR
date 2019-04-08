# -*- coding: UTF-8 -*-
from model import Point2SR
from utils import *
import time
import os
import configparser
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = configparser.ConfigParser()

config.read("base_config.ini")
NUM_PTS = config["hyperparameters"].getint("num_pts")
EPOCH = config["hyperparameters"].getint("epoch")
LR_D = config["hyperparameters"].getfloat("lr_d")
LR_G = config["hyperparameters"].getfloat("lr_g")
CAT_DIR = "./dataset1"

# Created without a result folder
if not os.path.exists("train_results"):
    os.mkdir("train_results")
# Set to local time
train_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
train_dir = os.path.join("train_results", train_time)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
# Copy the contents of the file from base_config.ini to config.ini
shutil.copyfile("base_config.ini", os.path.join(train_dir, "config.ini"))


def main():
    # train one model for every category
    # os.listdir: Returns a list of files and folders under the specified path
    for cat in os.listdir(CAT_DIR):
        if cat in os.listdir(train_dir):
            continue
        train_cat_dir = os.path.join(train_dir, cat)
        if not os.path.exists(train_cat_dir):
            os.mkdir(train_cat_dir)
        flog = open(os.path.join(train_cat_dir, 'log.txt'), 'w')

        # load training data from hdf5
        coordinates, ncoordinates, ocoordinates = load_single_cat_h5(cat, NUM_PTS, "train", "coordinates",
                                                                     "ncoordinates", "ocoordinates")
        printout(flog, "Loading training data! {}".format(cat))
        # load test data from hdf5
        test_coordinates, test_ncoordinates, test_ocoordinates = load_single_cat_h5(cat, NUM_PTS, "test", "coordinates",
                                                                                    "ncoordinates", "ocoordinates")

        printout(flog, "Loading test data! {}".format(cat))

        if test_coordinates.shape[0] > 8:
            # BATCH_SIZE = 8
            batch_size = 4
        elif test_coordinates.shape[0] > 4:
            # BATCH_SIZE = 4
            batch_size = 4
        else:
            # BATCH_SIZE = 2
            batch_size = 2

        with tf.Session() as sess:
            p2sr = Point2SR(sess, flog, num_pts=NUM_PTS, batch_size=batch_size, epoch=EPOCH, lr_g=LR_G, lr_d=LR_D)
            p2sr.train(train_cat_dir, coordinates, ncoordinates, ocoordinates,
                       test_coordinates, test_ncoordinates, test_ocoordinates)
        tf.reset_default_graph()


if __name__ == "__main__":
    main()
