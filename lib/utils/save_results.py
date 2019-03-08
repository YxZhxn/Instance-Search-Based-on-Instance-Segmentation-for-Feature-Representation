import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


def save_results(dstfn, im, detections, masks, class_names, cfg, scale=1.0):
    """
    save all detections from one image into diretory
    *-- written by Yx Zhxn --*
    :param dstfn: directory to save results
    :param im: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return: segmented image
    """
    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        dets = detections[j]
        msks = masks[j]
        cnt = 0
        for det, msk in zip(dets, msks):
            color = (random.random(), random.random(), random.random())  # generate a random color
            bbox = det[:4] * scale
            cod = bbox.astype(int)
            if im[cod[1]:cod[3], cod[0]:cod[2], 0].size > 0:
                msk = cv2.resize(msk, im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, 0].T.shape)
                bimsk = msk >= cfg.BINARY_THRESH
                bimsk = bimsk.astype(int)
                bimsk = np.repeat(bimsk[:, :, np.newaxis], 3, axis=2)
                mskd = im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] * bimsk
                clmsk = np.ones(bimsk.shape) * bimsk
                clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
                clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
                clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
                im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] = im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] + 0.8 * clmsk - 0.8 * mskd
                #--------------------------------------------
                # added to draw rectangles
                rect = plt.Rectangle((cod[0], cod[1]), cod[2] - cod[0], cod[3] - cod[1], linewidth=2.5,
                                     edgecolor=color, fill=False)
                plt.gca().add_patch(rect)
                # --------------------------------------------
            score = det[-1]
            cnt += 1
            # plt.gca().text((bbox[2] + bbox[0]) / 2, bbox[1], '{:s} {:.3f}'.format(name + '_' + str(cnt), score),
            #                bbox=dict(facecolor=color, alpha=0.9), fontsize=8, color='white')
            # plt.gca().text((bbox[2] + bbox[0]) / 2, bbox[1], '{:s}'.format(name + '_' + str(cnt)),
            #                bbox=dict(facecolor=color, alpha=0.9), fontsize=8, color='white')
    plt.imshow(im)
    plt.savefig(dstfn, bbox_inches='tight', pad_inches=0.0)
    return im