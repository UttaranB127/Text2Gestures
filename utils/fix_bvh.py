import torch
from utils.mocap_dataset import MocapDataset as MD

anim = MD.load_bvh('/media/uttaran/repo0/Gamma/MotionSim/src/quater_long_term_emonet_2/render/bvh/edin/test/000001.bvh')
animation_pred = {
    'joint_names': anim[0],
    'joint_offsets': torch.from_numpy(anim[2][1:]).unsqueeze(0),
    'joint_parents': anim[1],
    'positions': torch.from_numpy(anim[3]).unsqueeze(0),
    'rotations': torch.from_numpy(anim[4][1:]).unsqueeze(0)
}
MD.save_as_bvh(animation_pred,
               dataset_name='edin',
               subset_name='fixed')
