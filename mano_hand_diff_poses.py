import os
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
import copy
import config
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from wrappers import ModelPipeline
from utils import *


def mano_hand():

  generated_imgs = "./mano_hands/"

  o3d.visualization.RenderOption.line_width = 0.0
  view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
  window_size = 1080

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
  mesh.paint_uniform_color(config.HAND_COLOR)
  mesh.compute_triangle_normals()
  mesh.compute_vertex_normals()

  viewer = o3d.visualization.Visualizer()
  viewer.create_window(
    width=window_size + 1, height=window_size + 1,
    window_name='Minimal Hand - output'
  )
  viewer.add_geometry(mesh, reset_bounding_box=False)

  # yy: set camera
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

  # yy: rendering
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
  # frame_large = image
  # frame_large = np.flip(frame_large, axis=1).copy()
  # frame = imresize(frame_large, (128, 128))
  #
  # joint_locations, theta_mpii = model.process(frame)  # yy: joint locations & joint rotations
  # theta_mano = mpii_to_mano(theta_mpii)  # yy: shape: (21, 4)

  # v, joint_xyz = hand_mesh.set_abs_quat(theta_mano)
  # v *= 2 # for better visualization
  # v = v * 1000 + np.array([0, 0, 400])
  # v = mesh_smoother.process(v)
  # mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)  # (1538, 3)
  # mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)  # (778, 3)
  # mesh.paint_uniform_color(config.HAND_COLOR)
  # mesh.compute_triangle_normals()
  # mesh.compute_vertex_normals()


  # yy: add mesh back, show the minimal hand.
  viewer.poll_events()
  viewer.capture_screen_image(os.path.join(generated_imgs, 'mano_pose1.png'))


  # display.blit(
  #   pygame.surfarray.make_surface(
  #     np.transpose(
  #       imresize(frame_large, (window_size, window_size)
  #     ), (1, 0, 2))
  #   ),
  #   (0, 0)
  # )


  # pygame.display.update()
  # pygame.image.save(display, os.path.join(generated_imgs, 'mano_pose1_1.png'))

  mat = o3d.visualization.rendering.MaterialRecord()
  mat.shader = "unlitLine"
  mat.line_width = 1.0  # note that this is scaled with respect to pixels,


if __name__ == '__main__':
    mano_hand()
