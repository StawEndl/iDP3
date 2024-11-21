import json,os,cv2,zarr
import numpy as np
 


imgs = []
actions = []
states = []
episode_ends = []
json_files = ['/home/ace/codeM/Improved-3D-Diffusion-Policy-main/bowl-blue-grey0/data.json']

for json_file in json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    for da in data:
        img_path = da['imgs'].replace('/home/robot/DATASET/0929/bowl/','/home/ace/codeM/Improved-3D-Diffusion-Policy-main/')
        # os.system(f"cp -rf {img_path} /home/ace/codeM/Improved-3D-Diffusion-Policy-main/train_data_test/img/")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        
        action = [0]*25
        action[3+5+5:3+5+5+8] = da['pose']
        actions.append(action)
        
        state = [0]*32
        state[6+5+2+5+2:6+5+2+5+2+8] = da['pose']
        states.append(state)
        
        
        # if len(imgs)>5:
        #     break
    episode_ends.append(len(imgs))


with zarr.open("sample", mode='w') as zf:
    data_group = zf.create_group("data")
    imgs = np.array(imgs).astype(np.uint8)
    data_group.create_dataset("img", data=imgs, dtype='uint8')
    actions = np.array(actions).astype(np.float32)
    data_group.create_dataset("action", data=actions, dtype='float32')
    point_cloud = np.ones((len(imgs), 10000, 6)).astype(np.float32)
    data_group.create_dataset("point_cloud", data=point_cloud, dtype='float32')
    states = np.array(states).astype(np.float32)
    data_group.create_dataset("state", data=states, dtype='float32')
    
    data_group = zf.create_group("meta")
    episode_ends = np.array(episode_ends).astype(np.int64)
    data_group.create_dataset("episode_ends", data=episode_ends, dtype='int64')

