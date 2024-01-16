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

    # parser.add_argument("sino_path", nargs="?", default="data/90_sino.nii", help="Image to input")
    parser.add_argument("--config", nargs="?", default="config.json", help="JSON config for model")

    args = parser.parse_args()
    return args


class SVCT:
    def __init__(self, args):
        self.args = args
        config_path = self.args.config
        with open(config_path) as (config_file):
            self.config = json.load(config_file)

    def learning_the_implicit_function(self):
        config = self.config
        sv_sino_in_path = config["file"]["sv_sino_in_path"]
        dv_sino_out_path = config["file"]["dv_sino_out_path"]
        model_path = config["file"]["model_path"]
        sv_views = config["file"]["num_sv"]
        dv_views = config["file"]["num_dv"]
        sample_points_num = config["file"]["L"]
        lr = config["train"]["lr"]
        epochs = config["train"]["epoch"]
        summary_epoch = config["train"]["summary_epoch"]
        sample_N = config["train"]["sample_N"]
        batch_size = config["train"]["batch_size"]
        train_data = DataLoader(SinogramDataset(sample_points_num=sample_points_num, views=sv_views, sample_N=sample_N,
                                                sino_path=sv_sino_in_path, train=True), batch_size=batch_size,
                                shuffle=True)
        device = torch.device('cuda:{}'.format(str(config["train"]["gpu"]) if torch.cuda.is_available() else 'cpu'))
        model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1, encoding_config=config["encoding"],
                                              network_config=config["network"]).to(device)
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        model.train()
        loop = tqdm(range(epochs), leave=True)
        for epoch in loop:
            train_loss = 0
            batches = len(train_data)
            for batch, (ray, intensity) in enumerate(train_data):
                grid = ray.shape[2]
                ray = ray.ravel().reshape(-1, 2).to(device)
                intensity = intensity.to(device)
                pred = model(ray)
                pred = pred.reshape(batch_size, sample_N, grid)
                pred = torch.sum(pred, dim=2, keepdim=False)  # summation operate
                loss = loss_fn(pred, intensity)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss
            scheduler.step()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=train_loss.item() / batches, lr=scheduler.get_last_lr())
            if (epoch + 1) % summary_epoch == 0:
                torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    model = SVCT(get_args())
    model.learning_the_implicit_function()
