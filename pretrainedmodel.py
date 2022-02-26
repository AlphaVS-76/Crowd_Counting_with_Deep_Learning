import cv2
import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, name='scale_4', checkpoint_path=None, output_downscale=2,
                 PRED_DOWNSCALE_FACTORS=(8, 4, 2, 1), GAMMA=(1, 1, 2, 4), NUM_BOXES_PER_SCALE=3):

        super(CNN, self).__init__()
        self.name = name
        if torch.cuda.is_available():
            self.rgb_means = torch.cuda.FloatTensor([104.008, 116.669, 122.675])
        else:
            self.rgb_means = torch.FloatTensor([104.008, 116.669, 122.675])
        self.rgb_means = torch.autograd.Variable(self.rgb_means, requires_grad=False).unsqueeze(0).unsqueeze(
            2).unsqueeze(3)

        self.BOXES, self.BOX_SIZE_BINS = compute_boxes_and_sizes(PRED_DOWNSCALE_FACTORS, GAMMA, NUM_BOXES_PER_SCALE)
        self.output_downscale = output_downscale

        in_channels = 3
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.convA_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convA_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convA_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convA_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convA_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convB_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convB_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convB_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convB_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convB_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convC_1 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.convC_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convC_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convC_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convC_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convD_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convD_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convD_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convD_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convD_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.conv_before_transpose_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.transpose_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_1_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.transpose_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.transpose_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.transpose_4_1_a = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.transpose_4_1_b = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.transpose_4_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_4_2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.transpose_4_3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.conv_middle_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_middle_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_middle_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_mid_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.conv_lowest_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_lowest_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_lowest_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_lowest_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.conv_scale1_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_scale1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_scale1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x):
        mean_sub_input = x
        mean_sub_input -= self.rgb_means

        #################### Stage 1 ##########################

        main_out_block1 = self.relu(self.conv1_2(self.relu(self.conv1_1(mean_sub_input))))
        main_out_pool1 = self.pool1(main_out_block1)

        main_out_block2 = self.relu(self.conv2_2(self.relu(self.conv2_1(main_out_pool1))))
        main_out_pool2 = self.pool2(main_out_block2)

        main_out_block3 = self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(main_out_pool2))))))
        main_out_pool3 = self.pool3(main_out_block3)

        main_out_block4 = self.relu(self.conv4_3(self.relu(self.conv4_2(self.relu(self.conv4_1(main_out_pool3))))))
        main_out_pool4 = self.pool3(main_out_block4)

        main_out_block5 = self.relu(self.conv_before_transpose_1(
            self.relu(self.conv5_3(self.relu(self.conv5_2(self.relu(self.conv5_1(main_out_pool4))))))))

        main_out_rest = self.convA_5(self.relu(
            self.convA_4(self.relu(self.convA_3(self.relu(self.convA_2(self.relu(self.convA_1(main_out_block5)))))))))
        if self.name == "scale_1":
            return main_out_rest
        ################## Stage 2 ############################

        sub1_out_conv1 = self.relu(self.conv_mid_4(self.relu(
            self.conv_middle_3(self.relu(self.conv_middle_2(self.relu(self.conv_middle_1(main_out_pool3))))))))
        sub1_transpose = self.relu(self.transpose_1(main_out_block5))
        sub1_after_transpose_1 = self.relu(self.conv_after_transpose_1_1(sub1_transpose))

        sub1_concat = torch.cat((sub1_out_conv1, sub1_after_transpose_1), dim=1)

        sub1_out_rest = self.convB_5(self.relu(
            self.convB_4(self.relu(self.convB_3(self.relu(self.convB_2(self.relu(self.convB_1(sub1_concat)))))))))
        if self.name == "scale_2":
            return main_out_rest, sub1_out_rest
        ################# Stage 3 ############################

        sub2_out_conv1 = self.relu(self.conv_lowest_4(self.relu(
            self.conv_lowest_3(self.relu(self.conv_lowest_2(self.relu(self.conv_lowest_1(main_out_pool2))))))))
        sub2_transpose = self.relu(self.transpose_2(sub1_out_conv1))
        sub2_after_transpose_1 = self.relu(self.conv_after_transpose_2_1(sub2_transpose))

        sub3_transpose = self.relu(self.transpose_3(main_out_block5))
        sub3_after_transpose_1 = self.relu(self.conv_after_transpose_3_1(sub3_transpose))

        sub2_concat = torch.cat((sub2_out_conv1, sub2_after_transpose_1, sub3_after_transpose_1), dim=1)

        sub2_out_rest = self.convC_5(self.relu(
            self.convC_4(self.relu(self.convC_3(self.relu(self.convC_2(self.relu(self.convC_1(sub2_concat)))))))))

        if self.name == "scale_3":
            return main_out_rest, sub1_out_rest, sub2_out_rest

        ################# Stage 4 ############################
        sub4_out_conv1 = self.relu(
            self.conv_scale1_3(self.relu(self.conv_scale1_2(self.relu(self.conv_scale1_1(main_out_pool1))))))

        # TDF 1
        tdf_4_1_a = self.relu(self.transpose_4_1_a(main_out_block5))
        tdf_4_1_b = self.relu(self.transpose_4_1_b(tdf_4_1_a))
        after_tdf_4_1 = self.relu(self.conv_after_transpose_4_1(tdf_4_1_b))

        # TDF 2
        tdf_4_2 = self.relu(self.transpose_4_2(sub1_out_conv1))
        after_tdf_4_2 = self.relu(self.conv_after_transpose_4_2(tdf_4_2))

        # TDF 3
        tdf_4_3 = self.relu(self.transpose_4_3(sub2_out_conv1))
        after_tdf_4_3 = self.relu(self.conv_after_transpose_4_3(tdf_4_3))

        sub4_concat = torch.cat((sub4_out_conv1, after_tdf_4_1, after_tdf_4_2, after_tdf_4_3), dim=1)
        sub4_out_rest = self.convD_5(self.relu(
            self.convD_4(self.relu(self.convD_3(self.relu(self.convD_2(self.relu(self.convD_1(sub4_concat)))))))))

        if self.name == "scale_4":
            return main_out_rest, sub1_out_rest, sub2_out_rest, sub4_out_rest

    def head_detection(self, image, nms_thresh=0.25, thickness=2, multi_colours=True):
        if image.shape[0] % 16 or image.shape[1] % 16:
            image = cv2.resize(image, (image.shape[1]//16*16, image.shape[0]//16*16))
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(img_tensor.cuda())
        out = get_upsample_output(out, self.output_downscale)
        pred_dot_map, pred_box_map = get_box_and_dot_maps(out, nms_thresh, self.BOXES)
        img_out = get_boxed_img(image, pred_box_map, pred_box_map, pred_dot_map, self.output_downscale,
                                self.BOXES, self.BOX_SIZE_BINS, thickness=thickness, multi_colours=multi_colours)
        return pred_dot_map, img_out


import cv2
import torch
import numpy as np


def compute_boxes_and_sizes(PRED_DOWNSCALE_FACTORS, GAMMA, NUM_BOXES_PER_SCALE):

    BOX_SIZE_BINS = [1]
    g_idx = 0
    while len(BOX_SIZE_BINS) < NUM_BOXES_PER_SCALE * len(PRED_DOWNSCALE_FACTORS):
        gamma_idx = len(BOX_SIZE_BINS) // (len(GAMMA) - 1)
        box_size = BOX_SIZE_BINS[g_idx] + GAMMA[gamma_idx]
        BOX_SIZE_BINS.append(box_size)
        g_idx += 1

    BOX_SIZE_BINS_NPY = np.array(BOX_SIZE_BINS)
    BOXES = np.reshape(BOX_SIZE_BINS_NPY, (4, 3))
    BOXES = BOXES[::-1]

    return BOXES, BOX_SIZE_BINS


def upsample_single(input_, factor=2):
    channels = input_.size(1)
    indices = torch.nonzero(input_)
    indices_up = indices.clone()
    # Corner case!
    if indices_up.size(0) == 0:
        return torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor).cuda()
    indices_up[:, 2] *= factor
    indices_up[:, 3] *= factor

    output = torch.zeros(input_.size(0),input_.size(1), input_.size(2)*factor, input_.size(3)*factor).cuda()
    output[indices_up[:, 0], indices_up[:, 1], indices_up[:, 2], indices_up[:, 3]] = input_[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

    output[indices_up[:, 0], channels-1, indices_up[:, 2]+1, indices_up[:, 3]] = 1.0
    output[indices_up[:, 0], channels-1, indices_up[:, 2], indices_up[:, 3]+1] = 1.0
    output[indices_up[:, 0], channels-1, indices_up[:, 2]+1, indices_up[:, 3]+1] = 1.0

    # output_check = nn.functional.max_pool2d(output, kernel_size=2)

    return output


def get_upsample_output(model_output, output_downscale):
    upsample_max = int(np.log2(16 // output_downscale))
    upsample_pred = []
    for idx, out in enumerate(model_output):
        out = torch.nn.functional.softmax(out, dim=1)
        upsample_out = out
        for n in range(upsample_max - idx):
            upsample_out = upsample_single(upsample_out, factor=2)
        upsample_pred.append(upsample_out.cpu().data.numpy().squeeze(0))
    return upsample_pred


def box_NMS(predictions, nms_thresh, BOXES):
    Scores = []
    Boxes = []
    for k in range(len(BOXES)):
        scores = np.max(predictions[k], axis=0)
        boxes = np.argmax(predictions[k], axis=0)
        # index the boxes with BOXES to get h_map and w_map (both are the same for us)
        mask = (boxes < 3)  # removing Z
        boxes = (boxes + 1) * mask
        scores = (scores * mask)  # + 100 # added 100 since we take logsoftmax and it's negative!!

        boxes = (boxes == 1) * BOXES[k][0] + (boxes == 2) * BOXES[k][1] + (boxes == 3) * BOXES[k][2]
        Scores.append(scores)
        Boxes.append(boxes)

    x, y, h, w, scores = apply_nms(Scores, Boxes, Boxes, 0.5, thresh=nms_thresh)

    nms_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2]))  # since predictions[0] is of size 4 x H x W
    box_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2]))  # since predictions[0] is of size 4 x H x W
    for (xx, yy, hh) in zip(x, y, h):
        nms_out[yy, xx] = 1
        box_out[yy, xx] = hh

    assert (np.count_nonzero(nms_out) == len(x))

    return nms_out, box_out


def get_box_and_dot_maps(pred, nms_thresh, BOXES):
    assert (len(pred) == 4)
    # NMS on the multi-scale outputs
    nms_out, h = box_NMS(pred, nms_thresh, BOXES)
    return nms_out, h


def get_boxed_img(image, h_map, w_map, gt_pred_map, prediction_downscale, BOXES, BOX_SIZE_BINS,
                  thickness=1, multi_colours=False):
    if multi_colours:
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # colours for [1/8, 1/4, 1/2] scales

    if image.shape[2] != 3:
        boxed_img = image.astype(np.uint8).transpose((1, 2, 0)).copy()
    else:
        boxed_img = image.astype(np.uint8).copy()
    head_idx = np.where(gt_pred_map > 0)

    H, W = boxed_img.shape[:2]

    Y, X = head_idx[-2] , head_idx[-1]
    for y, x in zip(Y, X):

        h, w = h_map[y, x]*prediction_downscale, w_map[y, x]*prediction_downscale

        if multi_colours:
            selected_colour = colours[(BOX_SIZE_BINS.index(h // prediction_downscale)) // 3]
        else:
            selected_colour = (0, 255, 0)
        if h//2 in BOXES[3] or h//2 in BOXES[2]:
            t = 1
        else:
            t = thickness
        cv2.rectangle(boxed_img, (max(int(prediction_downscale * x - w / 2), 0), max(int(prediction_downscale * y - h / 2), 0)),
                      (min(int(prediction_downscale * x + w - w / 2), W), min(int(prediction_downscale * y + h - h / 2), H)), selected_colour, t)
    return boxed_img#.transpose((2, 0, 1))


import numpy as np


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        remove_index_1 = np.where(areas[i] == inter)
        remove_index_2 = np.where(areas[order[1:]] == inter)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ovr[remove_index_1] = 1.0
        ovr[remove_index_2] = 1.0
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def extract_conf_points(confidence_map, hmap):
    nms_conf_map = np.zeros_like(confidence_map[0])
    nms_conf_box = np.zeros_like(confidence_map[0])

    idx_1 = np.where(np.logical_and(confidence_map[0] > 0, confidence_map[1] <= 0))
    idx_2 = np.where(np.logical_and(confidence_map[0] <= 0, confidence_map[1] > 0))
    idx_common = np.where(np.logical_and(confidence_map[0] > 0, confidence_map[1] > 0))

    nms_conf_map[idx_1] = confidence_map[0][idx_1]
    nms_conf_map[idx_2] = confidence_map[1][idx_2]

    nms_conf_box[idx_1] = hmap[0][idx_1]
    nms_conf_box[idx_2] = hmap[1][idx_2]

    for ii in range(len(idx_common[0])):
        x, y = idx_common[0][ii], idx_common[1][ii]
        if confidence_map[0][x, y] > confidence_map[1][x, y]:
            nms_conf_map[x, y] = confidence_map[0][x, y]
            nms_conf_box[x, y] = hmap[0][x, y]
        else:
            nms_conf_map[x, y] = confidence_map[1][x, y]
            nms_conf_box[x, y] = hmap[1][x, y]

    assert (np.sum(nms_conf_map > 0) == len(idx_1[0]) + len(idx_2[0]) + len(idx_common[0]))

    return nms_conf_map, nms_conf_box


def apply_nms(confidence_map, hmap, wmap, dotmap_pred_downscale=2, thresh=0.3):
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[0], confidence_map[1]], [hmap[0], hmap[1]])
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[2], nms_conf_map], [hmap[2], nms_conf_box])
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[3], nms_conf_map], [hmap[3], nms_conf_box])

    confidence_map = nms_conf_map
    hmap = nms_conf_box
    wmap = nms_conf_box

    confidence_map = np.squeeze(confidence_map)
    hmap = np.squeeze(hmap)
    wmap = np.squeeze(wmap)

    dets_idx = np.where(confidence_map > 0)

    y, x = dets_idx[-2], dets_idx[-1]
    h, w = hmap[dets_idx], wmap[dets_idx]
    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    scores = confidence_map[dets_idx]

    dets = np.stack([np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores)], axis=1)
    # List of indices to keep
    keep = nms(dets, thresh)

    y, x = dets_idx[-2], dets_idx[-1]
    h, w = hmap[dets_idx], wmap[dets_idx]
    x = x[keep]
    y = y[keep]
    h = h[keep]
    w = w[keep]

    scores = scores[keep]
    return x, y, h, w, scores
    
