import cv2
import os
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = os.listdir('./datasets/one_hand/')
max_image_count = 11
print(IMAGE_FILES[0])

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 0.6
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 1
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  img_count = 0
  for idx, file in enumerate(IMAGE_FILES):
    

    if(img_count == max_image_count):
      break
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(f'./datasets/one_hand/{file}'), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      # print(idx)
      continue
    
    # print(len(results.multi_hand_landmarks))
    img_count += 1
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    j = 0

    points = np.load(f'./mh_one_hand/{idx}_joint_locations.npy', allow_pickle=True)
    hand_landmarks = results.multi_hand_landmarks[0]
    # print(hand_landmarks)

    max_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x 
    max_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    max_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
    
    min_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    min_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    min_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z 

    for hand_landmarks in results.multi_hand_landmarks:
      for landmark in mp_hands.HandLandmark:
        max_x = max(max_x, hand_landmarks.landmark[landmark].x)
        max_y = max(max_y, hand_landmarks.landmark[landmark].y)
        max_z = max(max_z, hand_landmarks.landmark[landmark].z)

        min_x = min(min_x, hand_landmarks.landmark[landmark].x)
        min_y = min(min_y, hand_landmarks.landmark[landmark].y)
        min_z = min(min_z, hand_landmarks.landmark[landmark].z)   

    # print(points[:, 0])
    # if(idx == 10):
    #   print(points[:,0])
    points[:, 0] = 1 - points[:, 0]
    points[:, 0] = (points[:, 0] - np.min(points[:, 0]))/(np.max(points[:, 0]) - np.min(points[:, 0]))
    points[:, 0] = points[:, 0]*(max_x - min_x) + min_x   
    points[:, 1] = 1 - points[:, 1]
    points[:, 1] = (points[:, 1] - np.min(points[:, 1]))/(np.max(points[:, 1]) - np.min(points[:, 1]))
    points[:, 1] = points[:, 1]*(max_y - min_y) + min_y  
    points[:, 2] = (points[:, 2] - np.min(points[:, 2]))/(np.max(points[:, 2]) - np.min(points[:, 2]))
    points[:, 2] = points[:, 2]*(max_z - min_z) + min_z   
    # print(points[:, 0])
    # print(max_x - min_x)
    dict_index = {0: 0, 1: 17, 2: 18, 3: 19, 4: 20, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8, 13: 9, 14: 10,
                  15:11, 16: 12, 17: 13, 18: 14, 19: 15, 20: 16}
    points1 = points.copy()
    for k in range(0, 21):
      points1[k][0] = points[dict_index[k]][0]
      points1[k][1] = points[dict_index[k]][1]
      points1[k][2] = points[dict_index[k]][2]
      # print(dict_index[k])

    points = points1
    # print(points[:, 0])

    for hand_landmarks in results.multi_hand_landmarks:
      # if j <= 1:
      #   j += 1
      #   continue
      # print('hand_landmarks:', mp_hands.HandLandmark)
    #   print(
    #       f'Index finger tip coordinates: (',
    #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
    #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
    #   )
      fingertips = [None] * 21
      
      c = 0
      for landmark in mp_hands.HandLandmark:
        # print(hand_landmarks.landmark[landmark].x)
        # print(landmark)
        hand_landmarks.landmark[landmark].x = points[c][0]
        hand_landmarks.landmark[landmark].y = points[c][1]
        hand_landmarks.landmark[landmark].z = points[c][2]
        # print(hand_landmarks.landmark[landmark].x)

        fingertips[c] = [hand_landmarks.landmark[landmark].x,  
                         hand_landmarks.landmark[landmark].y,
                         hand_landmarks.landmark[landmark].z]
        c += 1
        # print(fingertips[c - 1])

      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    
      c = 0 
      for landmark in mp_hands.HandLandmark:
        org = (int(255*hand_landmarks.landmark[landmark].x), int(255*hand_landmarks.landmark[landmark].y))
        # print(org)
        annotated_image = cv2.putText(annotated_image, str(c), org, font, fontScale, color, thickness, cv2.LINE_AA, False)
        # print(fingertips[c])
        c += 1
      
      np.set_printoptions(suppress=True)
      if(idx == 8):
        print(np.array(fingertips))
    # cv2.imwrite(
    #     './One_Hand_mediapipe/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    cv2.imwrite(
        './dataset_additional_info_one_hand/' + str(img_count - 1) + '_annotated_image_mh.png', cv2.flip(annotated_image, 1))
    np.save('./One_Hand_mediapipe/fingertips' + str(idx) + '_mh.npy', np.asarray(fingertips))
    
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    # for hand_world_landmarks in results.multi_hand_world_landmarks:
    #   mp_drawing.plot_landmarks(
    #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)