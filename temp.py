# # import os

# # img_folder_pth = './datasets/manual_train/' 
# # label_folder = './datasets/hands/'

# # print(os.listdir(img_folder_pth))

# # for file in os.listdir(img_folder_pth):
# #     if '.jpg' in file:
# #         os.system(f'copy {img_folder_pth}{file} ./datasets/hands/')

# #     if '.json' in file:
# #         os.system(f'copy {img_folder_pth}{file} ./datasets/hand_keypoints/')
# from __future__ import division
# import cv2
# import time
# import numpy as np

# protoFile = "./model/hand_points/pose_deploy.prototxt"
# weightsFile = "./model/hand_points/pose_iter_102000.caffemodel"
# nPoints = 22
# POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# frame = cv2.imread("./datasets/hands/009929335_01_r.jpg")
# frameCopy = np.copy(frame)
# frameWidth = frame.shape[1]
# frameHeight = frame.shape[0]
# aspect_ratio = frameWidth/frameHeight

# threshold = 0.1

# t = time.time()
# # input image dimensions for the network
# inHeight = 256
# inWidth = int(((aspect_ratio*inHeight)*8)//8)
# inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

# net.setInput(inpBlob)

# output = net.forward()
# print("time taken by network : {:.3f}".format(time.time() - t))

# # Empty list to store the detected keypoints
# points = []

# for i in range(nPoints):
#     # confidence map of corresponding body's part.
#     probMap = output[0, i, :, :]
#     probMap = cv2.resize(probMap, (frameWidth, frameHeight))

#     # Find global maxima of the probMap.
#     minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

#     if prob > threshold :
#         cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
#         cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

#         # Add the point to the list if the probability is greater than the threshold
#         points.append((int(point[0]), int(point[1])))
#     else :
#         points.append(None)

# # Draw Skeleton
# for pair in POSE_PAIRS:
#     partA = pair[0]
#     partB = pair[1]

#     if points[partA] and points[partB]:
#         cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
#         cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
#         cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


# cv2.imshow('Output-Keypoints', frameCopy)
# cv2.imshow('Output-Skeleton', frame)


# cv2.imwrite('Output-Keypoints.jpg', frameCopy)
# cv2.imwrite('Output-Skeleton.jpg', frame)

# print("Total time taken : {:.3f}".format(time.time() - t))

import os

# src_folder_pth = './dataset_additional_info1_filtered/' 
src_folder_pth = '.\\dataset_change_bg_256_left_right_3_channels\\'

print(os.listdir(src_folder_pth))

for file in os.listdir(src_folder_pth):
    print(f'rm -rf {src_folder_pth}{file}')
    if 'real_img' not in file:
        os.system(f'del {src_folder_pth}{file}')

