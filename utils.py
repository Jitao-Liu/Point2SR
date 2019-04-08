# -*- coding: UTF-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import csv
from PIL import Image, ImageChops

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def display(pts1, pts2, name_=None):
    dpi = 300
    pix_h = 1000
    pix_w = 2000
    marker_size = 5

    x1 = pts1[:, 0]
    y1 = pts1[:, 2]
    z1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 2]
    z2 = pts2[:, 1]
    max_range1 = np.array([x1.max() - x1.min(), y1.max()-y1.min(), z1.max()-z1.min()]).max() / 2.0
    max_range2 = np.array([x2.max() - x2.min(), y2.max() - y2.min(), z2.max() - z2.min()]).max() / 2.0
    mid_x1 = (x1.max() + x1.min()) * 0.5
    mid_y1 = (y1.max() + y1.min()) * 0.5
    mid_z1 = (z1.max() + z1.min()) * 0.5

    mid_x2 = (x2.max() + x2.min()) * 0.5
    mid_y2 = (y2.max() + y2.min()) * 0.5
    mid_z2 = (z2.max() + z2.min()) * 0.5

    fig = plt.figure()
    fig.set_size_inches(pix_w/dpi, pix_h/dpi)
    plt.subplots_adjust(top=1.2, bottom=-0.2, right=1.5, left=-0.5, hspace=0, wspace=-0.7)
    plt.margins(0, 0)
    ax = fig.add_subplot(121, projection='3d')
    bx = fig.add_subplot(122, projection='3d')
    ax.scatter(x1, y1, z1, edgecolors="none", c='#808080', s=marker_size, depthshade=True)
    bx.scatter(x2, y2, z2, edgecolors="none", c='#808080', s=marker_size, depthshade=True)
    ax.set_xlim(mid_x1 - max_range1, mid_x1 + max_range1)
    ax.set_ylim(mid_y1 - max_range1, mid_y1 + max_range1)
    ax.set_zlim(mid_z1 - max_range1, mid_z1 + max_range1)
    bx.set_xlim(mid_x2 - max_range2, mid_x2 + max_range2)
    bx.set_ylim(mid_y2 - max_range2, mid_y2 + max_range2)
    bx.set_zlim(mid_z2 - max_range2, mid_z2 + max_range2)
    # bx.patch.set_alpha(0)
    ax.set_aspect("equal")
    ax.grid("off")
    bx.set_aspect("equal")
    bx.grid("off")
    ax.axis('off')
    bx.axis("off")
    plt.axis('off')
    if name_:
        plt.savefig(name_, transparent=True, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


def display_only(pts, name_=None):
    dpi = 300
    pix_h = 1000
    pix_w = 1000
    marker_size = 5
    x = pts[:, 0]
    y = pts[:, 2]
    z = pts[:, 1]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    fig = plt.figure()
    fig.set_size_inches(pix_w/dpi, pix_h/dpi)
    plt.subplots_adjust(top=1.2, bottom=-0.2, right=1.5, left=-0.5, hspace=0, wspace=-0.7)
    plt.margins(0, 0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='#808080', edgecolors="none", s=marker_size, depthshade=True)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_aspect("equal")
    ax.grid("off")
    plt.axis('off')
    if name_:
        plt.savefig(name_, transparent=True, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


# print h5
def load_h5(path, *kwd):
    f = h5py.File(path)
    list_ = []
    for item in kwd:
        list_.append(f[item][:])
        print("{0} of shape {1} loaded!".format(item, f[item][:].shape))
        if item == "normalized coordinates" or item == "coordinates":
            pass
        if item == "color":
            print("color is of type {}".format(f[item][:].dtype))
    return list_


# load h5
def load_single_cat_h5(cat, num_pts, type, *kwd):
    path_ = os.path.join("./dataset", cat, "{0}_{1}".format(cat, num_pts), "{0}_{1}.h5".format(cat, type))
    print(path_)
    return load_h5(path_, *kwd)


# output and save to file
def printout(flog, data):
    print(data)
    flog.write(data + '\n')


def save_to_csv(data, nums, name_):
    out = open(name_, 'a', newline='')
    for num in range(nums.shape[1]):
        csv_write = csv.writer(out, dialect='excel')
        num_ = np.reshape(num, [1, ])
        data_ = {}
        data_[num] = np.hstack([num_, data[num]])
        csv_write.writerow(data_[num])


def trim_white_space(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -5)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def horizontal_concatnate_pic(fout, *fnames):
    images = [trim_white_space(Image.open(i).convert('RGB')) for i in fnames]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(fout)
