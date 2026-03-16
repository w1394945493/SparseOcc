import os
from tqdm import tqdm
import pickle
import mmcv
import warnings
warnings.filterwarnings("ignore")


# SparseOcc中使用的标注没有"scene_name'对应的键值对，补全key："scene_name"
if __name__=='__main__':
    ann_root='/c20250502/wangyushen/Datasets/NuScenes/method/sparseocc'
    ann_list = ['train','val']
    occ_gt_root='/c20250502/wangyushen/Datasets/occ3d_nuscenes/gts'
    save_dir = '/c20250502/wangyushen/Datasets/NuScenes/method/sparseocc/new'
    
    token2scene = {}
    scene_name_list = os.listdir(occ_gt_root)
    for scene_name in tqdm(scene_name_list):
        dir_path = os.path.join(occ_gt_root,scene_name)
        token_list = os.listdir(dir_path)
        for token in token_list:
            token2scene[token] = scene_name

    for ann in ann_list:
        ann_file = os.path.join(ann_root,f'/nuscenes_infos_{ann}_sweep.pkl')
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))

        for i in tqdm(range(len(data_infos))):
            token = data_infos[i]['token']
            data_infos[i]['scene_name'] = token2scene[token]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, os.path.basename(ann_file))
        data['infos'] = data_infos

        print(f"Saving to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    print("Success!")

