import os
from argparse import ArgumentParser

from utils.config_utils import read_config
from munch import munchify

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--config", type=str)
args = parser.parse_args()
config_path = args.config
args = read_config(config_path)
# set visible devices
device_list = args.device_list
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in device_list)

import torch
import torch.multiprocessing as mp

from arguments import DatasetParams, MapParams, OptimizationParams
from scene import Dataset
from SLAM.multiprocess.system import *
from SLAM.multiprocess.mapper import *
from SLAM.utils import *
from utils.general_utils import safe_state
from gui.gui_utils import ParamsGUI

torch.set_printoptions(4, sci_mode=False)
np.set_printoptions(4)
mp.set_sharing_strategy("file_system")


def main():
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser, sentinel=True)
    map_params = MapParams(parser)

    safe_state(args.quiet)
    optimization_params = optimization_params.extract(args)
    dataset_params = dataset_params.extract(args)
    map_params = map_params.extract(args)

    pipeline_params = munchify(args.pipeline_params)
    model_params = munchify(args.model_params)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    params_gui = ParamsGUI(
                    pipe=pipeline_params,
                    background=background
                )

    # Initialize dataset
    dataset = Dataset(
        dataset_params,
        shuffle=False,
        resolution_scales=dataset_params.resolution_scales,
    )

    # need to use spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    slam = SLAM(map_params, optimization_params, dataset, params_gui, args)
    slam.run()


if __name__ == "__main__":
    main()