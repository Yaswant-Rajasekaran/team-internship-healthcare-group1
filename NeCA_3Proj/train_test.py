import torch
import numpy as np
import argparse
import os
from src.config.configloading import load_config
from src.render import run_network
from src.trainer import Trainer
import pickle
from src.dataset import TIGREDataset as Dataset

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/CCTA.yaml",
                        help="configs file path")
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

        self.l2_loss = torch.nn.MSELoss(reduction='mean')

    # def set_directory(self, f_index):
    #     self.expdir = osp.join(cfg["exp"]["expdir"], str(f_index))
    #     self.ckptdir = osp.join(self.expdir, "ckpt.tar")
    #     # self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
    #     self.evaldir = osp.join(self.expdir, "eval")
    #     os.makedirs(self.evaldir, exist_ok=True)

    def set_data(self, proj, f_index):
        super().set_data(proj, f_index)
        
    # def set_data(self, proj, f_index):
    #     data={'projections':proj}
    #     self.dataconfig = data
    #     self.train_dset = Dataset(data, device)
    #     self.set_directory(f_index)
        
    def compute_loss(self, data, global_step, idx_epoch):
        loss = {"loss": 0.}

        projs = data.projs #.reshape(-1)
        # print(f'+++++++++++++++++++++++++ shape of proj is {projs.shape} ++++++++++++++++++++++++++++++++')
        # with torch.cuda.amp.autocast():#Imani
        image_pred = run_network(self.voxels, self.net, self.netchunk)
        # print(f'+++++++++++++++++++++++++ shape of image pred is {image_pred.shape} ++++++++++++++++++++++++++++++++')
        train_output = image_pred.squeeze()[None, ...] #.transpose(1,4).squeeze(4)

        train_projs_one = self.ct_projector_first.forward_project(train_output)
        train_projs_two = self.ct_projector_second.forward_project(train_output)
        train_projs_three = self.ct_projector_third.forward_project(train_output)
        
        train_projs = torch.cat((train_projs_one,train_projs_two , train_projs_three), 1)

        loss["loss"] = self.l2_loss(train_projs, projs)
        # print(f'+++++++++++++++++++++++++ shape of proj is {train_projs.shape} and {projs.shape} ++++++++++++++++++++++++++++++++')
        return loss

filepath='data/projections_odl_test.pkl'
with open(filepath, 'rb') as f:
    loaded_data = pickle.load(f)

projections=loaded_data['projections']
file_index=loaded_data['file_index']

done_files=np.array([file_index[0]])
done_file_path='./logs/done_files.npy'
np.save(done_file_path, done_files)

done_files=np.load(done_file_path)
print(f'done files {done_files}')

for i in range(1, len(file_index)):
    if(file_index[i] in done_files):
        continue

    print(f'----------------------- start training patient {file_index[i]} ------------------------------------')
    trainer = BasicTrainer()
    trainer.set_data(projections[i], file_index[i])
    trainer.start()

    done_files=np.vstack((done_files, file_index[i]))
    np.save(done_file_path, done_files)

