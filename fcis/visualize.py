import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def load_image(src, gt):
    """

    :param src:
    :param gt: ALOV format
    :return:
    """
    assert os.path.exists(src)
    assert os.path.exists(gt)

    im = cv2.imread(src)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    with open(gt, 'r') as strm:
        cod = strm.readline()
        cod = [int(float(ele)) for ele in cod.split(' ')[1:]]

    print cod

    color = (random.random(), random.random(), random.random())
    if cod[0] > cod[2]:
        roi = im[cod[3]:cod[7], cod[2]:cod[6]]
        rect = plt.Rectangle((cod[2], cod[3]), cod[6]-cod[2], cod[7]-cod[3], linewidth=1.5, edgecolor=color, fill=False)
    else:
        roi = im[cod[1]:cod[5], cod[0]:cod[4]]
        rect = plt.Rectangle((cod[0], cod[1]), cod[4] - cod[0], cod[5] - cod[1], linewidth=1.5, edgecolor=color, fill=False)
    plt.subplot(211).add_patch(rect)
    plt.subplot(211).imshow(im)
    plt.subplot(212).imshow(roi)
    plt.show()


def batch_load_image_otb(src_dir, gt_dir, dtc_dir):
    """

    :param src_dir:
    :param gt_dir: OTB format
    :return:
    """
    imgs = os.listdir(src_dir)
    imgs.sort()
    for img in imgs:

        fn = img[:-4].split('_')[0]
        with open(os.path.join(gt_dir, fn + '.txt'), 'r') as strm:
            cod = strm.readline()
            cod = [int(ele) for ele in cod.split('\t')[1:]]

        assert os.path.exists(src_dir + img), ('%s does not exist'.format(src_dir + img))
        im = cv2.imread(src_dir + img)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # roi = im[cod[1]:cod[1] + cod[3], cod[0]:cod[0] + cod[2]]
        roi = cv2.imread(os.path.join(dtc_dir, img[:-4] + '.dtc.jpg'))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        color = (random.random(), random.random(), random.random())  # generate a random color
        rect = plt.Rectangle((cod[0], cod[1]), cod[2], cod[3], linewidth=2.5, edgecolor=color, fill=False)

        plt.axis('off')
        plt.gcf().set_size_inches(12, 8)

        plt.subplot(211).axis('off')
        plt.subplot(211).add_patch(rect)
        plt.subplot(211).imshow(im)
        plt.subplot(211).set_title(fn)

        plt.subplot(212).axis('off')
        plt.subplot(212).imshow(roi)

        plt.tight_layout()
        plt.show()

def batch_load_image_alov(src_dir, gt_dir, dtc_dir):
    """

    :param src_dir:
    :param gt_dir: OTB format
    :return:
    """
    imgs = os.listdir(src_dir)
    imgs.sort()
    for img in imgs:

        fn = img[:-4].split('_')[0] + '_' + img[:-4].split('_')[1]
        with open(os.path.join(gt_dir, fn + '.ann'), 'r') as strm:
            cod = strm.readline()
            cod = [int(ele) for ele in cod.split('\t')[1:]]

        assert os.path.exists(src_dir + img), ('%s does not exist'.format(src_dir + img))
        im = cv2.imread(src_dir + img)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # roi = im[cod[1]:cod[1] + cod[3], cod[0]:cod[0] + cod[2]]
        roi = cv2.imread(os.path.join(dtc_dir, img[:-4] + '.dtc.jpg'))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        color = (random.random(), random.random(), random.random())  # generate a random color
        rect = plt.Rectangle((cod[0], cod[1]), cod[2], cod[3], linewidth=2.5, edgecolor=color, fill=False)

        plt.axis('off')
        plt.gcf().set_size_inches(12, 8)

        plt.subplot(211).axis('off')
        plt.subplot(211).add_patch(rect)
        plt.subplot(211).imshow(im)
        plt.subplot(211).set_title(fn)

        plt.subplot(212).axis('off')
        plt.subplot(212).imshow(roi)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    """"""
    src_dir = '/home/yz/cde/fcis/input/Instance/People/qry/'
    gt_dir = '/home/yz/cde/fcis/input/Instance/People/bbox/'
    dtc_dir = '/home/yz/cde/fcis/output/fcis/Instance/People/qry/'
    batch_load_image_otb(src_dir=src_dir, gt_dir=gt_dir, dtc_dir=dtc_dir)
    """"""

    """
    src = '/home/yz/cde/fcis/input/alov/00000001.jpg'
    gt = '/home/yz/cde/fcis/input/alov/14-LongDuration_video00001.ann'
    # src = '/home/yz/cde/fcis/input/alov/00000091.jpg'
    # gt = '/home/yz/cde/fcis/input/alov/12-MovingCamera_video00012.ann'
    load_image(src, gt)
    """

    """
    src_dir = '/home/yz/cde/fcis/input/ALOV/qry/'
    gt_dir = '/home/yz/cde/fcis/input/ALOV/bbox/'
    dtc_dir = '/home/yz/cde/fcis/output/fcis/ALOV/qry/'
    batch_load_image_alov(src_dir=src_dir, gt_dir=gt_dir, dtc_dir=dtc_dir)
    """