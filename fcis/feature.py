import os
import cv2
import glob
import pickle
import pprint
import numpy as np
import mxnet as mx
from sklearn.preprocessing import normalize

import _init_paths
from symbols import *
from core.tester import Predictor, conv_detect
from config.config import config, update_config

from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from utils.pooling import pooling_delegator
from utils.load_model import load_param
from utils.show_masks import show_masks
from utils.show_boxes import show_boxes
from utils.image import resize, transform
from bbox.bbox_transform import clip_boxes
from utils.save_results import save_results
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting


os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

os.environ['MXNET_GPU_MEM_POOL_RESERVE'] = '90'


class Extractor:
    """
    @Task: Extract image feature from rois
    @Author: Yx Zhxn
    """
    def __init__(self, cfg, mdl, src, dst=None, img=None, ftr=None, plsz=(1, 1), cpu_mask_vote=False, pca=False):
        """

        :param cfg:
        :param mdl:
        :param src:
        :param dst:
        :param img:
        :param ftr:
        :param plsz:
        """
        self.config_dir = cfg
        self.model_dir = mdl
        self.src_dir = src
        self.dst_dir = dst
        self.img = img
        self.ftr = ftr
        self.data_names = ['data', 'im_info']
        self.label_name = []
        self.batch_size = 16
        self.if_dim = 7  # dimension of output info
        self.pool_size = plsz  # size of roi pooling
        self.multiple = True  # detect multiple instances from one image
        self.cpu_mask_vote = cpu_mask_vote
        self.mapping = pca
        # self.result = {}  # dictionary: key -> image name, value -> tuple of(feature list, info list)

        # 1
        update_config(self.config_dir)
        # pprint.pprint(config)
        if not self.mapping:
            self.pca = pickle.load(open(config.pca, 'rb'))

        sum = 0
        for i in range(len(config.feature)):
            if config.feature[i] == 2:
                sum += 256
            elif config.feature[i] == 3:
                sum += 512
            elif config.feature[i] == 4:
                sum += 1024
            elif config.feature[i] == 5:
                sum += 2048
            elif config.feature[i] == 6:
                sum += 1024
        self.ft_dim = sum * self.pool_size[0] * self.pool_size[0]

        # 2
        sym_instance = eval(config.symbol)()
        self.sym = sym_instance.get_symbol(config, is_train=False)

        # 3
        self.num_classes = 81
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic.light', 'fire.hydrant', 'stop.sign', 'parking.meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                        'handbag', 'tie',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports.ball', 'kite', 'baseball.bat',
                        'baseball.glove', 'skateboard', 'surfboard', 'tennis.racket', 'bottle', 'wine.glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot.dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted.plant', 'bed', 'dining.table',
                        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell.phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy.bear',
                        'hair.drier', 'toothbrush']

        # 4
        # if multi_path == False:
        #     self.img_list = os.listdir(self.src_dir)
        #     self.img_list.sort()
        #     self.batch_list = None
        # else:
        #     self.img_list = []
        #     temp_list = os.listdir(self.src_dir)
        #     for itm in temp_list:
        #         sub_src_dir = os.path.join(self.src_dir, itm)
        #         sub_img_list = os.listdir(sub_src_dir)
        #         sub_img_list.sort()
        #         self.img_list.extend(sub_img_list)
        #     self.batch_list = None

        self.img_list = [y for x in os.walk(self.src_dir) for y in glob.glob(os.path.join(x[0], "*.jpg"))]
        self.img_list.sort()
        self.batch_list = None

        # 5
        self.data = None
        self.ctx = None
        self.predictor = None

        # 6
        if self.img is not None and self.ftr is not None:
            self.nm_strm = open(self.img, 'a')
            if not self.mapping:
                self.nm_strm.write('%d\n' % self.if_dim)
            self.ft_strm = open(self.ftr, 'a')
            if not self.mapping:
                self.ft_strm.write('%d\n' % self.ft_dim)

    def batch_extract(self, multiple=True, gt_dir=None, epoch=0):
        """

        :param multiple:
        :param gt_dir:
        :return:
        """
        if len(self.img_list) % self.batch_size != 0:
            batch = len(self.img_list) / self.batch_size + 1
        else:
            batch = len(self.img_list) / self.batch_size

        for i in xrange(batch):

            if i < batch - 1:
                self.batch_list = self.img_list[i * self.batch_size:(i + 1) * self.batch_size]
            else:
                self.batch_list = self.img_list[i * self.batch_size:]

            print '\nMini-batch %d\t' % (i+1)

            tmp_data = []
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]

            tic()
            for img in self.batch_list:
                assert os.path.exists(img), ('%s does not exist.'.format(img))
                im = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
                im_tensor = transform(im, config.network.PIXEL_MEANS)
                # im_info: height, width, scale
                im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
                tmp_data.append({self.data_names[0]: im_tensor, self.data_names[1]: im_info})

            self.ctx = [int(i) for i in config.gpus.split(',')]
            self.data = [[mx.nd.array(tmp_data[i][name], mx.gpu(self.ctx[0]))
                          for name in self.data_names] for i in xrange(len(tmp_data))]

            max_data_shape = [[(
                self.data_names[0],
                (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))
            )]]
            provide_data = [[(k, v.shape) for k, v in zip(self.data_names, self.data[i])] for i in xrange(len(self.data))]
            provide_label = [None for i in xrange(len(self.data))]

            arg_params, aux_params = load_param(self.model_dir, epoch, process=True)

            self.predictor = Predictor(self.sym, self.data_names, self.label_name, context=[mx.gpu(self.ctx[0])],
                                       max_data_shapes=max_data_shape, provide_data=provide_data,
                                       provide_label=provide_label,
                                       arg_params=arg_params, aux_params=aux_params)
            print 'preparation: %.4fs' % toc()

            if i == 0:
                self.warmup()

            self.forward(multiple=multiple, gt_dir=gt_dir)

        self.cleaner()

    def warmup(self):
        """

        :return:
        """
        for i in xrange(2):
            data_batch = mx.io.DataBatch(data=[self.data[0]], label=[], pad=0, index=0,
                                         provide_data=[[(k, v.shape) for k, v in zip(self.data_names, self.data[0])]],
                                         provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

            _, _, _, _, _ = conv_detect(self.predictor, data_batch, self.data_names, scales, config)

    def forward(self, multiple=True, gt_dir=None):
        """

        :param multiple:
        :param gt_dir:
        :return:
        """

        self.multiple = multiple  # if multiple is False, gt_dir must be provided
        if not self.multiple:
            assert gt_dir is not None

        for idx, itm in enumerate(self.batch_list):

            itm = itm.split('/')[-1]

            data_batch = mx.io.DataBatch(data=[self.data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(self.data_names, self.data[idx])]],
                                         provide_label=[None])

            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

            tic()

            scores, boxes, masks, convs, data_dict = conv_detect(self.predictor, data_batch,
                                                                 self.data_names, scales, config)
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')

            # return

            # (1) mask merge
            if not config.TEST.USE_MASK_MERGE:
                all_boxes = [[] for _ in xrange(self.num_classes)]
                all_masks = [[] for _ in xrange(self.num_classes)]
                nms = py_nms_wrapper(config.TEST.NMS)
                for j in range(1, self.num_classes):
                    indexes = np.where(scores[0][:, j] > 0.7)[0]
                    cls_scores = scores[0][indexes, j, np.newaxis]
                    cls_masks = masks[0][indexes, 1, :, :]
                    try:
                        if config.CLASS_AGNOSTIC:
                            cls_boxes = boxes[0][indexes, :]
                        else:
                            raise Exception()
                    except:
                        cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                    cls_dets = np.hstack((cls_boxes, cls_scores))
                    keep = nms(cls_dets)
                    all_boxes[j] = cls_dets[keep, :]
                    all_masks[j] = cls_masks[keep, :]
                dets = [all_boxes[j] for j in range(1, self.num_classes)]
                masks = [all_masks[j] for j in range(1, self.num_classes)]
            else:
                masks = masks[0][:, 1:, :, :]

                boxes = clip_boxes(boxes[0], (im_height, im_width))

                # gpu mask voting
                if not self.cpu_mask_vote:
                    result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], self.num_classes,
                                                                100, im_width, im_height,
                                                                config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                                config.BINARY_THRESH, self.ctx[0])

                # cpu mask voting
                else:
                    result_masks, result_dets = cpu_mask_voting(masks, boxes, scores[0], self.num_classes,
                                                                100, im_width, im_height,
                                                                config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                                config.BINARY_THRESH)

                # dets represent coordinates of bounding-boxes(up left, bottom right)
                dets = [result_dets[j] for j in range(1, self.num_classes)]
                masks = [result_masks[j][:, 0, :, :] for j in range(1, self.num_classes)]

            # (2) filter the result whose detection probability is under 0.7
            for i in xrange(len(dets)):
                keep = np.where(dets[i][:, -1] > 0.7)
                dets[i] = dets[i][keep]
                masks[i] = masks[i][keep]

            # (3) prepare for roi-pooling
            roi = []  # scaled bounding box coordinates(up left, bottom right)
            info = []  # class label, name of instance, probability, bounding box coordinates(up left, bottom right)
            if self.multiple:
                for k in xrange(0, self.num_classes - 1):
                    nums = len(dets[k])
                    if nums > 0:
                        for j in xrange(nums):
                            roi.append(dets[k][j][0:-1] * scales[0])  # note that the input image is scaled
                            info.append((k, itm[:-4] + '-' + self.classes[k] + '_' + str(j + 1), dets[k][j][-1],
                                         np.array(np.round(dets[k][j][0:-1]))))
            else:

                # Method 1
                dist = []
                temp_roi = []
                temp_info = []
                # Instance
                fn = itm[:-4].split('_')[0]
                # Instre
                # fn = itm[:-4]

                with open(os.path.join(gt_dir, fn + '.txt'), 'r') as strm:
                    cod = strm.readline()
                    if cod.split('\t')[-1] != "\n":
                        # Instance
                        cod = [int(ele) for ele in cod.split('\t')[1:]]
                    else:
                        # Instre
                        cod = [int(ele) for ele in cod.split('\t')[1:-1]]
                    cod_center_x = cod[0] + cod[2] / 2
                    cod_center_y = cod[1] + cod[3] / 2

                for k in xrange(0, self.num_classes - 1):
                    nums = len(dets[k])
                    if nums > 0:
                        for j in range(nums):
                            det_center_x = ((dets[k][j][2] - dets[k][j][0]) / 2 + dets[k][j][0])
                            det_center_y = ((dets[k][j][3] - dets[k][j][1]) / 2 + dets[k][j][1])
                            x = (det_center_x - cod_center_x) * (det_center_x - cod_center_x)
                            y = (det_center_y - cod_center_y) * (det_center_y - cod_center_y)
                            dist.append(np.sqrt(x + y))
                            temp_roi.append(dets[k][j][0:-1] * scales[0])  # note that the input image is scaled
                            temp_info.append((k, itm[:-4] + '-' + self.classes[k] + '_' + str(j + 1), dets[k][j][-1],
                                              np.array(np.round(dets[k][j][0:-1]))))

                npdist = np.array(dist)
                order = np.argsort(npdist)
                if len(order) > 0:
                    roi.append(temp_roi[order[0]])
                    info.append(temp_info[order[0]])

                # Method 2
                # cod[2] += cod[0]
                # cod[3] += cod[1]
                # cords = np.array(cod)
                # roi.append(cords * scales[0])
                # info.append((0, itm[:-4] + '-' + self.classes[0] + '_' + str(0 + 1), 0, np.array(cod)))

            # (4) perform roi-pooling & output
            features = pooling_delegator(convs, roi, self.pool_size, self.ctx[0], config)

            """ pca """
            if features is not None and not self.mapping:
                features = self.pca.transform(features)
                features = normalize(features, norm='l2', axis=1)

            if self.img is not None and self.ftr is not None and features is not None:
                self.output(info, features)

            print 'testing {} {:.4f}s'.format(itm, toc())

            # (5) visualize & save
            """"""
            if self.dst_dir:
                pass
                # im = cv2.imread(self.src_dir + itm)
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # im = show_masks(im, dets, masks, self.classes, config)
                # im = save_results(self.dst_dir + itm.replace('.jpg', '.dtc.jpg'), im, dets, masks, self.classes, config)
                # cv2.imwrite(self.dst_dir + itm.replace('.jpg', '.msk.jpg'), im)
            """"""

    def output(self, infos, features):
        """

        :param infos:
        :param features:
        :return:
        """
        cnt = len(infos)
        for i in xrange(cnt):
            info = infos[i]
            feature = features[i]
            self.nm_strm.write('%d %s %.4f %d %d %d %d\n' % (info[0], info[1], info[2], info[3][0], info[3][1],
                                                             info[3][2], info[3][3]))
            for j in xrange(feature.shape[0]):
                if j < feature.shape[0] - 1:
                    self.ft_strm.write('%f ' % feature[j])
                else:
                    self.ft_strm.write('%f\n' % feature[j])

    def cleaner(self):
        """

        :return:
        """
        if self.img is not None and self.ftr is not None:
            self.nm_strm.close()
            self.ft_strm.close()

    @staticmethod
    def test():
        """

        :return:
        """

        # original
        # cfg = '/home/yz/cde/fcis/experiments/fcis/cfgs/fcis_coco_demo.yaml'
        # mdl = '/home/yz/cde/fcis/model/fcis_coco'

        # FCISXD
        # cfg = '/home/yz/cde/fcis/experiments/fcis/cfgs/resnext_v1_101_coco_dc5_fcis_end2end_ohem.yaml'
        # mdl = '/home/yz/cde/fcis/output/resnext/coco-dc5/resnext_v1_101_coco_dc5_fcis_end2end_ohem/train2014_valminusminival2014/e2e'

        # FCISX
        cfg = '/home/yz/cde/fcis/experiments/fcis/cfgs/resnext_v1_101_coco_fcis_end2end_ohem.yaml'
        mdl = '/home/yz/cde/fcis/output/resnext/coco/resnext_v1_101_coco_fcis_end2end_ohem/train2014_valminusminival2014/e2e'

        # FCISD
        # cfg = '/home/yz/cde/fcis/experiments/fcis/cfgs/resnet_v1_101_coco_dc5_fcis_end2end_ohem.yaml'
        # mdl = '/home/yz/cde/fcis/output/dc5_fcis/coco/resnet_v1_101_coco_dc5_fcis_end2end_ohem/train2014_valminusminival2014/e2e'

        epoch = 8
        plsz = (1, 1)

        """
        Instance160
        """
        # QRY
        qry_src = '/home/yz/cde/fcis/input/Instance/All/qry/'
        qry_dst = '/home/yz/cde/fcis/output/fcis/Instance/All/qry/'
        qry_nm_fn = '/home/yz/cde/fcis/output/feature/Instance/qry/img1536-res34-fcisx.txt'
        qry_ft_fn = '/home/yz/cde/fcis/output/feature/Instance/qry/ftr1536-res34-fcisx.txt'
        gt = '/home/yz/cde/fcis/input/Instance/All/bbox/'
        extractor = Extractor(cfg=cfg, mdl=mdl, src=qry_src, dst=qry_dst, img=qry_nm_fn, ftr=qry_ft_fn, plsz=plsz, cpu_mask_vote=True)
        extractor.batch_extract(multiple=False, gt_dir=gt, epoch=epoch)

        # REF
        ref_src = '/home/yz/cde/fcis/input/Instance/All/ref/'
        ref_nm_fn = '/home/yz/cde/fcis/output/feature/Instance/ref/img1536-res34-fcisx.txt'
        ref_ft_fn = '/home/yz/cde/fcis/output/feature/Instance/ref/ftr1536-res34-fcisx.txt'
        extractor = Extractor(cfg=cfg, mdl=mdl, src=ref_src, img=ref_nm_fn, ftr=ref_ft_fn, plsz=plsz, cpu_mask_vote=True)
        extractor.batch_extract(epoch=epoch)

        # DIS
        # ref_src = '/home/yz/cde/fcis/input/Instance/All/dis1m/'
        # ref_nm_fn = '/home/yz/cde/fcis/output/feature/Instance/dis1m/img1536-res34-dis1m.txt'
        # ref_ft_fn = '/home/yz/cde/fcis/output/feature/Instance/dis1m/ftr1536-res34-dis1m.txt'
        # extractor = Extractor(cfg=cfg, mdl=mdl, src=ref_src, img=ref_nm_fn, ftr=ref_ft_fn, plsz=plsz)
        # extractor.batch_extract(epoch=epoch)


        """
        Instre
        """
        # # QRY
        # qry_src = '/home/yz/cde/fcis/input/Instre/qry/'
        # qry_dst = '/home/yz/cde/fcis/output/fcis/Instre/qry/'
        # qry_nm_fn = '/home/yz/cde/fcis/output/feature/Instre/qry/img1536-res34.txt'
        # qry_ft_fn = '/home/yz/cde/fcis/output/feature/Instre/qry/ftr1536-res34.txt'
        # gt = '/home/yz/cde/fcis/input/Instre/bbox/'
        # extractor = Extractor(cfg=cfg, mdl=mdl, src=qry_src, dst=qry_dst, img=qry_nm_fn, ftr=qry_ft_fn, plsz=plsz, cpu_mask_vote=True)
        # extractor.batch_extract(multiple=False, gt_dir=gt, epoch=epoch)
        #
        # # REF
        # ref_src = '/home/yz/cde/fcis/input/Instre/ref/'
        # ref_nm_fn = '/home/yz/cde/fcis/output/feature/Instre/ref/img1536-res34.txt'
        # ref_ft_fn = '/home/yz/cde/fcis/output/feature/Instre/ref/ftr1536-res34.txt'
        # extractor = Extractor(cfg=cfg, mdl=mdl, src=ref_src, img=ref_nm_fn, ftr=ref_ft_fn, plsz=plsz, cpu_mask_vote=True)
        # extractor.batch_extract(epoch=epoch)


        """ PCA """
        # src = '/home/yz/cde/fcis/input/pca/'
        # nm_fn = '/home/yz/cde/fcis/output/feature/pca/img-res34-fcisx.txt'
        # ft_fn = '/home/yz/cde/fcis/output/feature/pca/ftr-res34-fcisx.txt'
        # extractor = Extractor(cfg=cfg, mdl=mdl, src=src, img=nm_fn, ftr=ft_fn, plsz=plsz, pca=True)
        # extractor.batch_extract(epoch=epoch)


if __name__ == '__main__':
    Extractor.test()
