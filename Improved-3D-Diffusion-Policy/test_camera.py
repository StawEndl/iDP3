########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

def cloud2depth(pcd):
   
    # 转换为numpy数组
    points = np.asarray(pcd.points)
    
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
    
    # 计算深度图
    depth = o3d.geometry.Image(np.asarray(pcd.points[:, 2], dtype=np.float64))
    
    return depth

def write_depth():
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud("Pointcloud.ply")
    depth = cloud2depth(pcd)
    # 保存深度图
    o3d.io.write_image("depth.png", depth)

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")



def main():
    # print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    # init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
    #                              coordinate_units=sl.UNIT.METER,
    #                              coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    # parse_args(init)
    # # print(f"init: {init}")
    # zed = sl.Camera()
    # status = zed.open(init)
    # if status != sl.ERROR_CODE.SUCCESS:
    #     print(repr(status))
    #     exit()

    # res = sl.Resolution()
    # res.width = 224
    # res.height = 224

    # # camera_model = zed.get_camera_information().camera_model
    # # Create OpenGL viewer
    # # viewer = gl.GLViewer()
    # # print(f"sys.argv: {sys.argv}")  #sys.argv = ['Improved-3D-Diffusion-Policy/test_camera.py']
    # # viewer.init(1, sys.argv, camera_model, res)
    # image = sl.Mat(res.width, res.height)
    # depth = sl.Mat(res.width, res.height)
    # point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # # while viewer.is_available():
    # #     if zed.grab() == sl.ERROR_CODE.SUCCESS:
    # #         zed.retrieve_image(image, sl.VIEW.LEFT)
    # #         zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    # #         zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
    # #         print(f"???{image.numpy().shape()}, {depth.numpy().shape()}, {point_cloud.numpy().shape()}")
    # #         # print(f"????{type(point_cloud)}")
    # #         viewer.updateData(point_cloud)
    # #         import ipdb; ipdb.set_trace()
    # #         if(viewer.save_data == True):
    # #             point_cloud_to_save = sl.Mat()
    # #             zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
    # #             err = point_cloud_to_save.write('Pointcloud.ply')
    # #             if(err == sl.ERROR_CODE.SUCCESS):
    # #                 print("Current .ply file saving succeed")
    # #             else:
    # #                 print("Current .ply file failed")
    # #             viewer.save_data = False

    # if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #     zed.retrieve_image(image, sl.VIEW.LEFT)
    #     zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    #     zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
    #     print(f"???{image.numpy().shape}, {depth.numpy().shape}, {point_cloud.numpy().shape}")
    #     # point_cloud_to_save = sl.Mat()
    #     # zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
    #     # err = point_cloud_to_save.write('Pointcloud.ply')
    #     # if(err == sl.ERROR_CODE.SUCCESS):
    #     #     print("Current .ply file saving succeed")
    #     # else:
    #     #     print("Current .ply file failed")
    # zed.close()
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                                coordinate_units=sl.UNIT.METER,
                                coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    parse_args(init)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 720
    res.height = 404
    # import ipdb; ipdb.set_trace()

    # camera_model = zed.get_camera_information().camera_model
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # print(f"sys.argv: {sys.argv}")  #sys.argv = ['Improved-3D-Diffusion-Policy/test_camera.py']
    # viewer.init(1, sys.argv, camera_model, res)

    # point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    debug = False
    # while True:
    #     color_frame, depth_frame, point_cloud_frame = self.get_vision()
    #     self.queue.put([color_frame, depth_frame, point_cloud_frame])
    #     time.sleep(0.05)
    
    # while viewer.is_available():
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
        color_frame = image.numpy()
        depth_frame = None
        point_cloud_frame = None

        depth = sl.Mat()
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
        depth_frame = depth.numpy()
        print(depth_frame.shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 