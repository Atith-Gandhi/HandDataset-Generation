import cv2
import time
import numpy as np
from PIL import Image
import os
import json
import random


"""
Description: generate the dataset in preparation for the Generative model training
TODO: draw fingers far away first, then draw closer fingers
"""



# POSE_PAIRS = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
#                   [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]])

# CORRESPOND_PAIRS = {0: 0, 1: 5, 2: 6, 3: 7, 4: 9, 5: 10, 6: 11, 7: 17, 8: 18, 9: 19, 10: 13, 11: 14, 12: 15, 13: 1,
#                     14: 2, 15: 3, 16: 8, 17: 12, 18: 20, 19: 16, 20: 4}

# UP_DOWN_PARIS = np.array([[3, 4], [7, 8], [11, 12], [15, 16], [19, 20], [2, 3], [6, 7], [10, 11], [14, 15], [18, 19],
#                  [1, 2], [5, 6], [9, 10], [13, 14], [17, 18], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17]])

POSE_PAIRS = np.array([[0, 1], [1, 6], [6, 7], [7, 8], [0, 2], [2, 9], [9, 10], [10, 11], [0, 3], [3, 12], [12, 13],
                       [13, 14], [0, 4], [4, 15], [15, 16], [16, 17], [0, 5], [5, 18], [18, 19], [19, 20]])

UP_DOWN_PAIRS = np.array([[7, 8], [10, 11], [13, 14], [16, 17], [19, 20], [6, 7], [9, 10], [12, 13], [15, 16], [18, 19], 
                          [1, 6], [2, 9],[3, 12], [4, 15], [5, 18], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])

SMAPLE_NUM, THRE = 100, 0.00001
cnt = 0

img_folder_pth = './datasets/dataset_10000_256_left_right_no_wrists_fix_angle/rendered_hand/'  # yy: dataset path containing the rgb images
# keypoints_folder_pth = './dataset_1000/final_imgs/kp/'  # yy: keypoints path containing the keypoints info
colored_mh_pth = './datasets/dataset_10000_256_left_right_no_wrists_fix_angle/dataset_change_bg/'  # yy: path to the generated minimal hands (defined in generate_mh_info.py)
skeleton_pth = './datasets/dataset_10000_256_left_right_no_wrists_fix_angle/'  # yy: path to save the generated dataset
dataset_6_channels_pth = './dataset_10000_256_left_right_no_wrists_no_bg_6_channels/'
dataset_3_channels_pth = './dataset_10000_256_left_right_no_wrists_no_bg_3_channels/'
dataset_456_channels_pth = './dataset_10000_256_left_right_no_wrists_no_bg_456_channels/'
cubes_mh_pth = './datasets/dataset_10000_256_left_right_no_wrists_fix_angle/cubes/'
kp_pth = './datasets/dataset_10000_256_left_right_no_wrists_fix_angle/kp/'

for i, img in enumerate(sorted(os.listdir(img_folder_pth))[:SMAPLE_NUM]):
    # if(i< 7500):
    #     continue
    print(i)
    cnt = i
    rand_int = random.randint(0, 1000)
    # mh_img_kp_img_pth = f'{i:03d}' + ".png"
    # colored_mh_img_pth = f'{i:03d}' + ".png"
    mh_img_kp_img = cv2.imread(os.path.join(cubes_mh_pth, img))
    colored_mh_img = cv2.imread(os.path.join(colored_mh_pth, img))

    # print(np.asarray(mh_img_kp_img))

    # iterate kp img
    kp_positions = {}
    z_positions = {}
    mh_img_kp_img = mh_img_kp_img / 255
    # print(mh_img_kp_img.shape)
    mh_img_kp_joint_locations_path = kp_pth + img[: -4] + ".npy"
    joint_locations = np.load(mh_img_kp_joint_locations_path)
    # print(mh_img_kp_img.shape)
    unique_cube_values = np.sort(np.unique(mh_img_kp_img[:, :, 0]))

    for y in range(1, mh_img_kp_img.shape[0]):  # 432 434
        for x in range(1, mh_img_kp_img.shape[1]):  # 568  567
            # if (abs(mh_img_kp_img[y-1, x, 0] - 1.0) < THRE)\
            #     and (abs(mh_img_kp_img[y, x-1, 0] - 1.0) < THRE)\
            #     and (abs(mh_img_kp_img[y, x, 0] - 1.0) > THRE):
            # if(mh_img_kp_img[y, x, 0] != 1.0):
            #     print(mh_img_kp_img[y, x, 0])
            # print(y + 2)
            # try:
            # if (abs(mh_img_kp_img[y, x, 0] - 1.0) > THRE)\
            #     and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y + 1, x + 1, 0] \
            #     and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y + 2, x + 2, 0] \
            #     and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y + 1, x, 0] \
            #     and mh_img_kp_img[y, x, 0] == mh_img_kp_img[y, x + 1, 0] \
            #     and mh_img_kp_img[y, x, 0] != mh_img_kp_img[y - 1, x, 0] \
            #     and mh_img_kp_img[y, x, 0] != mh_img_kp_img[y, x - 1, 0]:
            if(mh_img_kp_img[y, x, 0] != 1.0):
                # idx = round(mh_img_kp_img[y, x, 0] / 0.04)
                # print(mh_img_kp_img[y, x, 0])
                idx = np.where(unique_cube_values == mh_img_kp_img[y, x, 0])
                idx = idx[0][0]
                if idx == 22:
                    a  = 1
                # print(idx)
                # a = 1
                # real_idx = CORRESPOND_PAIRS[idx]
                real_idx = idx
                # print(joint_locations.shape)
                # print(idx)
                kp_positions[real_idx] = (x, y)
                z_positions[real_idx] = joint_locations[int(idx)][2]
                # print(kp_positions[real_idx])
        
    # print(len(np.unique(mh_img_kp_img[:, :, 0])))
    # print(len(kp_positions))
    # print(len(kp_positions))
    if len(kp_positions) != 21:
        print("[Error] less than 21 keypoints detected!!")
        continue
    # else:
    #     print("yes")

    if('left' in img):
        colored_mh_img = colored_mh_img*0 + 100
    else:
        colored_mh_img = colored_mh_img*0 + 200
    points = [kp_positions[q] for q in range(21)]
    # print(points.shape)


    # calculate average color for each line
    pose_pair_joint_locations = np.array([z_positions[p[1]] + z_positions[p[0]] for p in POSE_PAIRS])
    pair_color_ls = [None] * 21
    # for i in np.flip(np.argsort(pose_pair_joint_locations)):
    for i in reversed(range(len(POSE_PAIRS))):
        pair = POSE_PAIRS[i]
        pt1 = list(points[int(pair[0])])
        pt2 = list(points[int(pair[1])])

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
        if(rgb[0] < 50.0):
            rgb = np.array([50.0, 50.0, 50.0])
        pair_color_ls[i] = rgb
        
        # print("i: ",i, "rgb: ", rgb)
    pair_color_ls = np.array(pair_color_ls)

    # color the img
    one_channel = colored_mh_img[:, :, -1:]
    img_c1, img_c2, img_c3 = np.zeros_like(one_channel), np.zeros_like(one_channel), np.zeros_like(one_channel)
    blur_level, THICKNESS = 5, 5
    color_c1 = [50, 150, 250, 100, 200]
    color_c2 = [100, 200, 50, 150, 250]

    # for i in np.flip(np.argsort(pose_pair_joint_locations)):
    for i in reversed(range(len(POSE_PAIRS))):
        pair = POSE_PAIRS[i]
        color = color_c1[i // 4]
        partA = pair[0]
        partB = pair[1]
        # print("i:",i, " color: ", color)

        if points[partA] and points[partB]:
            cv2.line(img_c1, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(img_c1, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(img_c1, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        img_c1 = cv2.blur(img_c1, (blur_level, blur_level))

    updown_pair_joint_locations = np.array([z_positions[p[1]] + z_positions[p[0]] for p in UP_DOWN_PAIRS])
    for i in np.argsort(updown_pair_joint_locations):
        pair = UP_DOWN_PAIRS[i]
        color = color_c2[i // 5]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c2, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(img_c2, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(img_c2, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        img_c2 = cv2.blur(img_c2, (blur_level, blur_level))

    # for i in np.flip(np.argsort(pose_pair_joint_locations)):
    for i in reversed(range(len(POSE_PAIRS))):
        pair = POSE_PAIRS[i]
        color = pair_color_ls[i][2]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c3, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(img_c3, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(img_c3, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        img_c3 = cv2.blur(img_c3, (blur_level, blur_level))


    # Draw skeletons on the minimal hand
    # for i in np.flip(np.argsort(pose_pair_joint_locations)):
    for i in reversed(range(len(POSE_PAIRS))):
        pair = POSE_PAIRS[i]
        color = pair_color_ls[i]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(colored_mh_img, points[partA], points[partB], color, THICKNESS * 2)
            cv2.circle(colored_mh_img, points[partA], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(colored_mh_img, points[partB], THICKNESS, color, thickness=-1, lineType=cv2.FILLED)

        # colored_mh_img = cv2.blur(colored_mh_img, (blur_level, blur_level))


    # stack 3 channels

    ch1_name = f'{cnt:06d}' + "_channel_1_new.jpg"
    ch_1_img = np.stack([img_c1, img_c1, img_c1], axis=2)
    # print(img_c1.shape)
    # ch_1_img = img_c1
    cv2.imwrite(skeleton_pth + ch1_name, ch_1_img)

    ch2_name = f'{cnt:06d}' + "_channel_2_new.jpg"
    ch_2_img = np.stack([img_c2, img_c2, img_c2], axis=2)
    cv2.imwrite(skeleton_pth + ch2_name, ch_2_img)

    ch3_name = f'{cnt:06d}' + "_channel_3_new.jpg"
    ch_3_img = np.stack([img_c3, img_c3, img_c3], axis=2)
    cv2.imwrite(skeleton_pth + ch3_name, ch_3_img)

    frame_mh_no_light_image_name = f'{cnt:06d}' + "_skeleton_on_colored_mh.jpg"
    cv2.imwrite(skeleton_pth + frame_mh_no_light_image_name, colored_mh_img.astype(np.uint8))


    # TODO
    ori_img = cv2.imread(os.path.join(img_folder_pth, img))
    # with open(os.path.join(keypoints_folder_pth, img.replace("jpg", "json"))) as f:
    #     data = json.load(f)
    #     pts = data['hand_pts']

    # for m, pair in enumerate(POSE_PAIRS):
    #     color = pair_color_ls[m]
    #     img = cv2.line(ori_img, tuple(np.array(pts[pair[0]][:2], dtype=np.int)),
    #                    tuple(np.array(pts[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)

    # Save the original real image
    real_image_with_skeleton_name = f'{cnt:06d}' + "_skeleton_real_img.jpg"
    cv2.imwrite(skeleton_pth + real_image_with_skeleton_name, ori_img)
    cv2.imwrite(dataset_3_channels_pth + real_image_with_skeleton_name, ori_img)



    # create 6 channel img
    np.random.seed(100)
    one_channel = ori_img[:, :, -1:]
    img_c1, img_c2, img_c3 = np.zeros_like(one_channel), np.zeros_like(one_channel), np.zeros_like(one_channel)
    blur_level, THICKNESS = 5, 5
    color_c1 = [50, 150, 250, 100, 200]
    color_c2 = [250, 50, 200, 150, 100]

    # np.random.shuffle(color_c1)
    # np.random.shuffle(color_c2)

    print(color_c1)
    print(color_c2)
    # for i in np.flip(np.argsort(pose_pair_joint_locations)):
    for i in reversed(range(len(POSE_PAIRS))):
        pair = POSE_PAIRS[i]
        color = color_c1[i // 4]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c1, tuple(np.array(points[pair[0]][:2], dtype=np.int)),
                     tuple(np.array(points[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
            cv2.circle(img_c1, tuple(np.array(points[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)
            cv2.circle(img_c1, tuple(np.array(points[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)

        img_c1 = cv2.blur(img_c1, (blur_level, blur_level))


    # for i in np.flip(np.argsort(updown_pair_joint_locations)):
    for i in reversed(range(len(UP_DOWN_PAIRS))):
        pair = UP_DOWN_PAIRS[i]
        color = color_c2[i // 5]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c2, tuple(np.array(points[pair[0]][:2], dtype=np.int)),
                     tuple(np.array(points[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
            cv2.circle(img_c2, tuple(np.array(points[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)
            cv2.circle(img_c2, tuple(np.array(points[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)

        img_c2 = cv2.blur(img_c2, (blur_level, blur_level))


    # for i in np.flip(np.argsort(pose_pair_joint_locations)):
    for i in reversed(range(len(POSE_PAIRS))):
        pair = POSE_PAIRS[i]
        color = pair_color_ls[i][2]
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img_c3, tuple(np.array(points[pair[0]][:2], dtype=np.int)),
                     tuple(np.array(points[pair[1]][:2], dtype=np.int)), color, THICKNESS * 2)
            cv2.circle(img_c3, tuple(np.array(points[pair[0]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)
            cv2.circle(img_c3, tuple(np.array(points[pair[1]][:2], dtype=np.int)), THICKNESS, color, thickness=-1,
                       lineType=cv2.FILLED)

        img_c3 = cv2.blur(img_c3, (blur_level, blur_level))


    # stack 6 channels
    
    # img_c1, img_c2, img_c3 = img_c1[:, :, None], img_c2[:, :, None], img_c3[:, :, None]
    img_c1, img_c2, img_c3, ori_img = Image.fromarray(img_c1), Image.fromarray(img_c2), Image.fromarray(img_c3), Image.fromarray(ori_img)
    # img_c1 = img_c1.resize((256, 256))
    # img_c2 = img_c2.resize((256, 256))
    # img_c3 = img_c3.resize((256, 256))
    # ori_img = ori_img.resize((256, 256))

    img_c1, img_c2, img_c3, ori_img = np.asarray(img_c1), np.asarray(img_c2), np.asarray(img_c3), np.asarray(ori_img) 

    # print(img_c1.shape)
    # print(ori_img.shape)

    # print(cnt)
    # need to change this
    img_final = np.stack([ori_img[:, :, 0], ori_img[:, :, 1], ori_img[:, :, 2], img_c1, img_c2, img_c3], axis=2)
    img_final_name = f'{cnt:06d}' + "_img_final.npy"
    np.save(skeleton_pth + img_final_name, img_final)
    np.save(dataset_6_channels_pth + img_final_name, img_final)

    img_final = np.stack([img_c1, img_c2, img_c3], axis=2)
    img_final_name = f'{cnt:06d}' + "_img_final.png"
    cv2.imwrite(dataset_456_channels_pth + img_final_name, img_final)
    # np.save(skeleton_pth + img_final_name, img_final)
    # np.save(dataset_6_channels_pth + img_final_name, img_final)

    # save each channel
    ch1_name = f'{cnt:06d}' + "_channel_1_new.jpg"
    ch_1_img = np.stack([img_c1, img_c1, img_c1], axis=2)
    # print(img_c1.shape)
    # ch_1_img = img_c1
    cv2.imwrite(skeleton_pth + ch1_name, ch_1_img)

    ch2_name = f'{cnt:06d}' + "_channel_2_new.jpg"
    ch_2_img = np.stack([img_c2, img_c2, img_c2], axis=2)
    cv2.imwrite(skeleton_pth + ch2_name, ch_2_img)

    ch3_name = f'{cnt:06d}' + "_channel_3_new.jpg"
    ch_3_img = np.stack([img_c3, img_c3, img_c3], axis=2)
    cv2.imwrite(skeleton_pth + ch3_name, ch_3_img)

    # cnt += 1