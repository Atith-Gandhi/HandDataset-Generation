import os
import random

import cv2
import keyboard
import numpy as np
import open3d
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
import copy
import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils import *
import time
import matplotlib.pyplot as plt
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering


def live_application(capture):
  """
  Launch an application that reads from a webcam and estimates hand pose at
  real-time.

  The captured hand must be the right hand, but will be flipped internally
  and rendered.

  Parameters
  ----------
  capture : object
    An object from `capture.py` to read capture stream from.
  """
  ############ output visualization ############
  view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
  window_size = 1080

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
  mesh.compute_vertex_normals()

  viewer = o3d.visualization.Visualizer()
  viewer.create_window(
    width=window_size + 1, height=window_size + 1,
    window_name='Minimal Hand - output'
  )
  viewer.add_geometry(mesh)

  view_control = viewer.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  extrinsic = cam_params.extrinsic.copy()
  extrinsic[0:3, 3] = 0
  cam_params.extrinsic = extrinsic
  cam_params.intrinsic.set_intrinsics(
    window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
    window_size // 2, window_size // 2
  )
  view_control.convert_from_pinhole_camera_parameters(cam_params)
  view_control.set_constant_z_far(1000)

  render_option = viewer.get_render_option()
  render_option.load_from_json('./render_option.json')
  viewer.update_renderer()

  ############ input visualization ############
  pygame.init()
  display = pygame.display.set_mode((window_size, window_size))
  pygame.display.set_caption('Minimal Hand - input')

  ############ misc ############
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  clock = pygame.time.Clock()
  model = ModelPipeline()

  while True:
    frame_large = capture.read()
    if frame_large is None:
      continue
    if frame_large.shape[0] > frame_large.shape[1]:
      margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
      frame_large = frame_large[margin:-margin]
    else:
      margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
      frame_large = frame_large[:, margin:-margin]

    frame_large = np.flip(frame_large, axis=1).copy()
    frame = imresize(frame_large, (128, 128))

    _, theta_mpii = model.process(frame)
    theta_mano = mpii_to_mano(theta_mpii)

    v = hand_mesh.set_abs_quat(theta_mano)
    v *= 2 # for better visualization
    v = v * 1000 + np.array([0, 0, 400])
    v = mesh_smoother.process(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry(mesh)

    viewer.poll_events()

    display.blit(
      pygame.surfarray.make_surface(
        np.transpose(
          imresize(frame_large, (window_size, window_size)
        ), (1, 0, 2))
      ),
      (0, 0)
    )
    pygame.display.update()

    if keyboard.is_pressed("esc"):
      break

    clock.tick(30)



def live_application_img(image, i):
  """
  Launch an application that reads from a webcam and estimates hand pose at
  real-time.

  The captured hand must be the right hand, but will be flipped internally
  and rendered.

  Parameters
  ----------
  capture : object
    An object from `capture.py` to read capture stream from.
  """
  generated_imgs = "./dec7_imgs/generated_imgs/"

  ############ output visualization ############
  view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
  window_size = 1080

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
  mesh.compute_vertex_normals()


  # # yy: add scene
  # gui.Application.instance.initialize()
  # window = gui.Application.instance.create_window('img', width=window_size + 1, height=window_size + 1)
  # widget = gui.SceneWidget()
  # widget.scene = rendering.Open3DScene(window.renderer)
  # widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0.577, -0.577, -0.577))
  # window.add_child(widget)




  viewer = o3d.visualization.Visualizer()
  viewer.create_window(
    width=window_size + 1, height=window_size + 1,
    window_name='Minimal Hand - output'
  )
  viewer.add_geometry(mesh)

  # scene = o3d.visualization.rendering.Scene()
  # scene.enable_indirect_light(False)

  # viewer.capture_screen_image(os.path.join(generated_imgs, str(i) + '_mesh.png'), True)
  # viewer.capture_screen_image('test.png', True)

  view_control = viewer.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  extrinsic = cam_params.extrinsic.copy()
  extrinsic[0:3, 3] = 0
  cam_params.extrinsic = extrinsic
  cam_params.intrinsic.set_intrinsics(
    window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
    window_size // 2, window_size // 2
  )
  view_control.convert_from_pinhole_camera_parameters(cam_params)
  view_control.set_constant_z_far(1000)

  render_option = viewer.get_render_option()
  render_option.load_from_json('./render_option.json')
  viewer.update_renderer()


  ############ input visualization ############
  pygame.init()
  display = pygame.display.set_mode((window_size, window_size))
  pygame.display.set_caption('Minimal Hand - input')

  ############ misc ############
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  clock = pygame.time.Clock()
  model = ModelPipeline()


  frame_large = image

  # if frame_large.shape[0] > frame_large.shape[1]:
  #   margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
  #   frame_large = frame_large[margin:-margin]
  # else:
  #   margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
  #   frame_large = frame_large[:, margin:-margin]

  frame_large = np.flip(frame_large, axis=1).copy()
  frame = imresize(frame_large, (128, 128))

  joint_locations, theta_mpii = model.process(frame)  # yy: joint locations & joint rotations
  theta_mano = mpii_to_mano(theta_mpii)  # yy: shape: (21, 4)

  v, joint_xyz = hand_mesh.set_abs_quat(theta_mano)
  v *= 2 # for better visualization
  v = v * 1000 + np.array([0, 0, 400])
  v = mesh_smoother.process(v)
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)  # (1538, 3)
  mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)  # (778, 3)
  mesh.paint_uniform_color(config.HAND_COLOR)
  mesh.compute_triangle_normals()
  mesh.compute_vertex_normals()





  # yy: draw cubes
  mesh_box = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
  mesh_box.compute_vertex_normals()
  mesh_box.paint_uniform_color([0., 0., 0.])
  # viewer.add_geometry(mesh_box)


  v = copy.deepcopy(joint_xyz)
  v *= 2  # for better visualization
  v = v * 1000 + np.array([0, 0, 400])  # old: [0, 0, 360]
  v = np.matmul(view_mat, v.T).T
  # # v = mesh_smoother.process(v)


  # color_ls = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
  #             np.array([1., 1., 0.]), np.array([0., 1., 1.]), np.array([1., 0., 1.]),
  #             np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
  #             np.array([1., 1., 0.]), np.array([0., 1., 1.]), np.array([1., 0., 1.]),
  #             np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
  #             np.array([1., 1., 0.]), np.array([0., 1., 1.]), np.array([1., 0., 1.]),
  #             np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
  #             ]
  for q in range(21):
    new_mesh = copy.deepcopy(mesh_box)
    viewer.add_geometry(new_mesh)
    new_mesh.paint_uniform_color(np.array([1., 1., 1.]) * 0.04 * q)  # 0.04
    new_mesh.translate(tuple(v[q, :]), relative=True)
    # new_mesh.paint_uniform_color(color_ls[q])  # 0.04
    viewer.update_geometry(new_mesh)



  # # yy: add color
  # mesh.vertex_colors[0] = np.array([1., 0.706, 1.])
  #
  # lgth = np.asarray(mesh.vertex_colors).shape[0]
  # unit = 1.0 / lgth
  #
  # for j in range(lgth):
  #   # # random
  #   # color = np.array([random.random() for _ in range(3)]).astype('float64')
  #   # gradually increase
  #   color = np.array([unit, unit, unit]) * j
  #   mesh.vertex_colors[j] = color

  def calc_angle(vec):
    return np.arcsin(np.abs(vec[2]) / np.linalg.norm(vec))

  # yy: add color by direction
  # ver_normals = np.asarray(mesh.vertex_normals)
  # up_idx = (ver_normals[:, 2] > 0).nonzero()[0]
  # side_idx = (np.apply_along_axis(calc_angle, 1, ver_normals) < (np.pi / 6)).nonzero()[0]
  # np.save("up_idx.npy", up_idx)
  # np.save("side_idx.npy", side_idx)

  up_idx = np.load("up_idx.npy")
  side_idx = np.load("side_idx.npy")
  lgth = np.asarray(mesh.vertex_colors).shape[0]
  for j in range(lgth):
    if j in up_idx:
      # mesh.vertex_colors[j] = np.array([0.6, 0.5, 0.4])
      mesh.vertex_colors[j] = np.array([0.1, 0.1, 0.1])
    else:
      # mesh.vertex_colors[j] = np.array([0.1, 0.2, 0.3])
      mesh.vertex_colors[j] = np.array([0.9, 0.9, 0.9])

    # if j in side_idx:
    #   mesh.vertex_colors[j] = np.array([0.9, 0.6, 0.8])
  # mesh.vertex_colors[ver_normals[:, 2] > 0, :] = np.array(0.6, 0.5, 0.4)
  # mesh.vertex_colors[ver_normals[:, 2] < 0, :] = np.array(0.1, 0.2, 0.3)



  # viewer.update_geometry(mesh)
  # viewer.poll_events()
  # viewer.capture_screen_image(os.path.join(generated_imgs, str(i) + '_5.png'))

  viewer.remove_geometry(mesh)
  viewer.poll_events()
  viewer.capture_screen_image(os.path.join(generated_imgs, str(i) + '_1.png'))

  viewer.add_geometry(mesh)
  viewer.poll_events()
  viewer.capture_screen_image(os.path.join(generated_imgs, str(i) + '_2.png'))


  display.blit(
    pygame.surfarray.make_surface(
      np.transpose(
        imresize(frame_large, (window_size, window_size)
      ), (1, 0, 2))
    ),
    (0, 0)
  )

  # clock.tick(1000)

  pygame.display.update()

  pygame.image.save(display, os.path.join(generated_imgs, str(i) + '_0.png'))

  # clock.tick(1000)



if __name__ == '__main__':
  img_folder_pth = "./datasets/CMU_dataset_5000_ori/"
  # img_folder_pth = "./dec7_imgs/ori_cmu_imgs_50"

  # img_folder_pth = '../datasets/CMU_dataset_5000_ori/'
  # keypoints_folder_pth = '../datasets/CMU_keypoints_5000/'


  # img_folder_pth = "./original_images/"
  # img_folder_pth = "./c/"
  # img_folder_pth = "../nov16_imgs/cmu_ori_imgs_10/"
  for i, img in enumerate(sorted(os.listdir(img_folder_pth))[:100]):
    print(img)
    img_path = os.path.join(img_folder_pth, img)
    img = cv2.imread(img_path)
    live_application_img(img, i)



    # live_application(OpenCVCapture())
