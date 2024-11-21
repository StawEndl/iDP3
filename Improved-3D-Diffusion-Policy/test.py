import h5py,cv2
import zarr
import numpy as np

def read_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        data = f['dataset_name'][:]  # 读取数据集
    return data


# data = read_h5_file("/storage/liujinxin/code/tram/Improved-3D-Diffusion-Policy-main/raw_data_example/0.h5")

print()

# with zarr.open("/home/ace/codeM/Improved-3D-Diffusion-Policy-main/sample") as zf:
#     print(zf.tree())
    
with zarr.open("/home/ace/codeM/Improved-3D-Diffusion-Policy-main/training_data_example") as zf:
    print(zf.tree())


all_img = zf['data/img']
all_point_cloud = zf['data/point_cloud']
all_episode_ends = zf['meta/episode_ends']

print(zf['data/action'].shape, all_episode_ends)

img_array = np.array(all_img)
print(len(img_array))
cv2.imwrite("280.png", img_array[280])
cv2.imwrite("281.png", img_array[281])
cv2.imwrite("282.png", img_array[282])
cv2.imwrite("283.png", img_array[283])
cv2.imwrite("284.png", img_array[284])
