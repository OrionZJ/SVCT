import argparse
import sys
import time

import numpy as np
import SimpleITK as sitk
import commentjson as json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SinogramDataset

try:
    import tinycudann as tcnn
except ImportError:
    print("This program requires the tiny-cuda-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-cuda-nn$ cd bindings/torch")
    print("tiny-cuda-nn/bindings/torch$ python setup.py install")
    print("============================================================")
    sys.exit()


def get_args():
    parser = argparse.ArgumentParser(description="Training model with PyTorch bindings.")

    parser.add_argument("-c", "--config", nargs="?", default="config.json", help="JSON config for model")

    args = parser.parse_args()
    return args


class SVCT:
    def __init__(self, args):
        self.args = args
        config_path = self.args.config
        with open(config_path) as (config_file):
            self.config = json.load(config_file)
        config = self.config
        self.sv_sino_in_path = config["file"]["sv_sino_in_path"]
        self.dv_sino_out_path = config["file"]["dv_sino_out_path"]
        self.model_path = config["file"]["model_path"]
        self.sv_views = config["file"]["num_sv"]
        self.dv_views = config["file"]["num_dv"]
        self.sample_points_num = config["file"]["L"]
        self.lr = config["train"]["lr"]
        self.epochs = config["train"]["epoch"]
        self.summary_epoch = config["train"]["summary_epoch"]
        self.sample_N = config["train"]["sample_N"]
        self.batch_size = config["train"]["batch_size"]
        self.device = torch.device(
            'cuda:{}'.format(str(self.config["train"]["gpu"]) if torch.cuda.is_available() else 'cpu'))

    def learning_the_implicit_function(self):
        train_data = DataLoader(
            SinogramDataset(sample_points_num=self.sample_points_num, views=self.sv_views, sample_N=self.sample_N,
                            sino_path=self.sv_sino_in_path, train=True), batch_size=self.batch_size,
            shuffle=True)

        model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1, encoding_config=self.config["encoding"],
                                              network_config=self.config["network"]).to(self.device)
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(params=(model.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        loop = tqdm(range(self.epochs), leave=True)
        for epoch in loop:
            model.train()
            train_loss = 0
            batches = len(train_data)
            for batch, (ray, intensity) in enumerate(train_data):
                grid = ray.shape[2]
                ray = ray.reshape(-1, 2).to(self.device)
                intensity = intensity.to(self.device)
                pred = model(ray)
                pred = pred.view(self.batch_size, self.sample_N, grid)
                pred = torch.sum(pred, dim=2)  # summation operate
                loss = loss_fn(pred, intensity)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            loop.set_description(f'Epoch: [{epoch + 1}/{self.epochs}]')
            loop.set_postfix(loss=train_loss / batches, lr=scheduler.get_last_lr()[0])
            if (epoch + 1) % self.summary_epoch == 0:
                torch.save(model.state_dict(), self.model_path)
                print("\nCheckpoint save to path: {}".format(self.model_path))
        torch.save(model.state_dict(), self.model_path)
        print("\nFinal model save to path: {}".format(self.model_path))

    def re_projection_reconstruction(self):
        sv_sinogram = sitk.GetArrayFromImage(sitk.ReadImage(self.sv_sino_in_path))
        test_data = DataLoader(
            SinogramDataset(sample_points_num=self.sample_points_num, views=self.dv_views, sample_N=self.sample_N,
                            sino_path=self.sv_sino_in_path, train=False), batch_size=self.batch_size,
            shuffle=False)
        model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1, encoding_config=self.config["encoding"],
                                              network_config=self.config["network"]).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        batches = len(test_data)
        intensities = []
        with torch.no_grad():
            loop = tqdm(enumerate(test_data), total=batches, leave=True)
            for batch, ray in loop:
                grid = ray.shape[2]
                ray = ray.reshape(-1, 2).to(self.device)
                pred = model(ray)
                pred = pred.reshape(-1, grid, grid)
                pred = torch.sum(pred, dim=2, keepdim=False)  # summation operate
                intensities.extend(pred.cpu().numpy().tolist())
                loop.set_description(f'Reconstructing views: [{(batch + 1) * 3}/{batches * 3}]')
        intensities = np.array(intensities)

        # projection swapping
        swap_idx = np.rint(np.linspace(0, self.dv_views - 1, self.sv_views))
        for i, idx in enumerate(swap_idx):
            intensities[int(idx)] = sv_sinogram[i]
        dv_sino = sitk.GetImageFromArray(intensities)
        save_path = self.dv_sino_out_path + "/%s_recon_sino.nii" % self.dv_views
        sitk.WriteImage(dv_sino, save_path)
        print("\nReconstructed sinogram save to path: {}".format(save_path))


if __name__ == "__main__":
    args = get_args()
    model = SVCT(args)

    model.learning_the_implicit_function()
    model.re_projection_reconstruction()
