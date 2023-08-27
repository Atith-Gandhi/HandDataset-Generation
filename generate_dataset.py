import cv2
import time
import numpy as np
from PIL import Image
import os
import json


"""
Description: generate the dataset in preparation for the Generative model training
TODO: draw fingers far away first, then draw closer fingers
"""



POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

CORRESPOND_PAIRS = {0: 0, 1: 5, 2: 6, 3: 7, 4: 9, 5: 10, 6: 11, 7: 17, 8: 18, 9: 19, 10: 13, 11: 14, 12: 15, 13: 1,
                    14: 2, 15: 3, 16: 8, 17: 12, 18: 20, 19: 16, 20: 4}

UP_DOWN_PARIS = [[3, 4], [7, 8], [11, 12], [15, 16], [19, 20], [2, 3], [6, 7], [10, 11], [14, 15], [18, 19],
                 [1, 2], [5, 6], [9, 10], [13, 14], [17, 18], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17]]


SMAPLE_NUM, THRE = 100, 0.00001
cnt = 0

img_folder_pth = './datasets/CMU_dataset_5000_ori/'  # yy: dataset path containing the rgb images
keypoints_folder_pth = './datasets/CMU_keypoints_5000/'  # yy: keypoints path containing the keypoints info
colored_mh_pth = './dec7_imgs/generated_imgs/'  # yy: path to the generated minimal hands (defined in generate_mh_info.py)
skeleton_pth = './dataset_additional_info1/'  # yy: path to save the generated dataset


for i, img in enumerate(sorted(os.listdir(img_folder_pth))[:SMAPLE_NUM]):
    mh_img_kp_img_pth = str(i) + "_1.png"
    colored_mh_img_pth = str(i) + "_2.png"
    mh_img_kp_img = cv2.imread(os.path.join(colored_mh_pth, mh_img_kp_img_pth))
    colored_mh_img = cv2.imread(os.path.join(colored_mh_pth, colored_mh_img_pth))

    # iterate kp img
    kp_positions = {}
    mh_img_kp_img = mh_img_kp_img / 255
    for y in range(1, mh_img_kp_img.shape[0]):  # 432 434
        for x in range(1, mh_img_kp_img.shape[1]):  # 568  567
            # if (abs(mh_img_kp_img[y-1, x, 0] - 1.0) < THRE)\
            #     and (abs(mh_img_kp_img[y, x-1, 0] - 1.0) < THRE)\
            #     and (abs(mh_img_kp_img[y, x, 0] - 1.0) > THRE):
            if (abs(mh_img_kp_img[y, x, 0] - 1.0) > THRE)\
                and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y + 1, x + 1, 0] \
                and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y + 2, x + 2, 0] \
                and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y + 1, x, 0] \
                and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y, x + 1, 0] \
                and mh_img_kp_img[y, x, 0] != mh_img_kp_img[y - 1, x, 0] \
                and mh_img_kp_img[y, x, 0] != mh_img_kp_img[y, x - 1, 0]:
                idx = round(mh_img_kp_img[y+2, x+2, 0] / 0.04)
                if idx == 22:
                    a  = 1
                # print(idx)
                # a = 1
                real_idx = CORRESPOND_PAIRS[idx]
                kp_positions[real_idx] = (x, y)

    if len(kp_positions) != 21:
        print("[Error] less than 21 keypoints detected!!")
        continue

    points = [kp_positions[q] for q in range(21)]


    # calculate average color for each line
    pair_color_ls = []
    for i, pair in enumerate(POSE_PAIRS):
        pt1 = list(points[pair[0]])
        pt2 = list(points[pair[1]])

        rgb_ls = []
        if pt1[0] == pt2[0]:
            x = pt1[0]
            for y in range(min(pt1[1], pt2[1]), max(pt1[1], pt2[1]) + 1):
                rgb_ls.append(colored_mh_img[int(y), int(x), :])
        elif pt1[1] == pt2[1]:
            y = pt1[1]
            for x in range(min(pt1[0], pt2[0]), max(pt1[0], pt2[0]) + 1):
                rgb_ls.append(colored_mh_img[int(y), int(x), :])
        elif abs(pt1[0] - pt2[0]) >= abs(pt1[1] - pt2[1]):
            if pt1[0] > pt2[0]:
                tmp = pt1.copy()
                pt1 = pt2.copy()
                pt2 = tmp.copy()
            for x in range(pt1[0], pt2[0] + 1):
                y = pt1[1] + ((pt2[1] - pt1[1]) / (pt2[0] - pt1[0])) * (x - pt1[0])
                rgb_ls.append(colored_mh_img[int(y), int(x), :])
        else:
            if pt1[1] > pt2[1]:
                tmp = pt1.copy()
                pt1 = pt2.copy()
                pt2 = tmp.copy()
            for y in range(pt1[1], pt2[1] + 1):
                x = pt1[0] + ((pt2[0] - pt1[0]) / (pt2[1] - pt1[1])) * (y - pt1[1])
                rgb_ls.append(colored_mh_img[int(y), int(x), :])
        rgb = np.mean(np.array(rgb_ls), axis=0)
        pair_color_ls.append(rgb)
    pair_color_ls = np.array(pair_color_ls)

    # color the img
    one_channel = colored_mh_img[:, :, -1:]
    img_c1, img_c2, img_c3 = np.zeros_like(one_channel), np.zeros_like(one_channel), np.zeros_like(one_channel)
    blur_level, THICKNESS = 5, 5
    color_c1 = [50, 100, 150, 200, 250]
    color_c2 = [50, 100, 150, 200, 250]
    for i, pair in enumerate(POSE_PAIRS):
        color = color_c1[i // 4]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c1, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(img_c1, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(img_c1, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        img_c1 = cv2.blur(img_c1, (blur_level, blur_level))

    for i, pair in enumerate(UP_DOWN_PARIS):
        color = color_c2[i // 5]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c2, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(img_c2, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(img_c2, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        img_c2 = cv2.blur(img_c2, (blur_level, blur_level))

    for i, pair in enumerate(POSE_PAIRS):
        color = pair_color_ls[i][2]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c3, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(img_c3, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(img_c3, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        img_c3 = cv2.blur(img_c3, (blur_level, blur_level))


    # Draw skeletons on the minimal hand
    for i, pair in enumerate(POSE_PAIRS):
        color = pair_color_ls[i]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(colored_mh_img, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(colored_mh_img, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(colored_mh_img, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        # colored_mh_img = cv2.blur(colored_mh_img, (blur_level, blur_level))


    # stack 3 channels

    ch1_name = str(cnt) + "_channel_1.jpg"
    ch_1_img = np.stack([img_c1, img_c1, img_c1], axis=2)
    cv2.imwrite(skeleton_pth + ch1_name, ch_1_img)

    ch2_name = str(cnt) + "_channel_2.jpg"
    ch_2_img = np.stack([img_c2, img_c2, img_c2], axis=2)
    cv2.imwrite(skeleton_pth + ch2_name, ch_2_img)

    ch3_name = str(cnt) + "_channel_3.jpg"
    ch_3_img = np.stack([img_c3, img_c3, img_c3], axis=2)
    cv2.imwrite(skeleton_pth + ch3_name, ch_3_img)

    frame_mh_no_light_image_name = str(cnt) + "_skeleton_on_colored_mh.jpg"
    cv2.imwrite(skeleton_pth + frame_mh_no_light_image_name, np.fliplr(colored_mh_img).astype(np.uint8))



    # TODO
    ori_img = cv2.imread(os.path.join(img_folder_pth, img))
    with open(os.path.join(keypoints_folder_pth, img.replace("jpg", "json"))) as f:
        data = json.load(f)
        pts = data['hand_pts']

    # for m, pair in enumerate(POSE_PAIRS):
    #     color = pair_color_ls[m]
    #     img = cv2.line(ori_img, tuple(np.array(pts[pair[0]][:2], dtype=np.int)),
    #                    tuple(np.array(pts[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)

    # Save the original real image
    real_image_with_skeleton_name = str(cnt) + "_skeleton_real_img.jpg"
    cv2.imwrite(skeleton_pth + real_image_with_skeleton_name, ori_img)



    # create 6 channel img
    one_channel = ori_img[:, :, -1:]
    img_c1, img_c2, img_c3 = np.zeros_like(one_channel), np.zeros_like(one_channel), np.zeros_like(one_channel)
    blur_level, THICKNESS = 5, 5
    color_c1 = [50, 100, 150, 200, 250]
    color_c2 = [50, 100, 150, 200, 250]
    for i, pair in enumerate(POSE_PAIRS):
        color = color_c1[i // 4]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c1, tuple(np.array(pts[pair[0]][:2], dtype=np.int)),
                     tuple(np.array(pts[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
            cv2.circle(img_c1, tuple(np.array(pts[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)
            cv2.circle(img_c1, tuple(np.array(pts[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)

        img_c1 = cv2.blur(img_c1, (blur_level, blur_level))


    for i, pair in enumerate(UP_DOWN_PARIS):
        color = color_c2[i // 5]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c2, tuple(np.array(pts[pair[0]][:2], dtype=np.int)),
                     tuple(np.array(pts[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
            cv2.circle(img_c2, tuple(np.array(pts[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)
            cv2.circle(img_c2, tuple(np.array(pts[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)

        img_c2 = cv2.blur(img_c2, (blur_level, blur_level))


    for i, pair in enumerate(POSE_PAIRS):
        color = pair_color_ls[i][2]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c3, tuple(np.array(pts[pair[0]][:2], dtype=np.int)),
                     tuple(np.array(pts[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
            cv2.circle(img_c3, tuple(np.array(pts[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)
            cv2.circle(img_c3, tuple(np.array(pts[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)

        img_c3 = cv2.blur(img_c3, (blur_level, blur_level))


    # stack 6 channels
    # img_c1, img_c2, img_c3 = img_c1[:, :, None], img_c2[:, :, None], img_c3[:, :, None]
    img_final = np.stack([ori_img[:, :, 0], ori_img[:, :, 1], ori_img[:, :, 2], img_c1, img_c2, img_c3], axis=2)
    img_final_name = str(cnt) + "_img_final.npy"
    np.save(skeleton_pth + img_final_name, img_final)


    # save each channel
    ch1_name = str(cnt) + "_channel_1.jpg"
    ch_1_img = np.stack([img_c1, img_c1, img_c1], axis=2)
    cv2.imwrite(skeleton_pth + ch1_name, ch_1_img)

    ch2_name = str(cnt) + "_channel_2.jpg"
    ch_2_img = np.stack([img_c2, img_c2, img_c2], axis=2)
    cv2.imwrite(skeleton_pth + ch2_name, ch_2_img)

    ch3_name = str(cnt) + "_channel_3.jpg"
    ch_3_img = np.stack([img_c3, img_c3, img_c3], axis=2)
    cv2.imwrite(skeleton_pth + ch3_name, ch_3_img)

    cnt += 1