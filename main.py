import json
import argparse
from trainer import train
import wandb
import os
from utils.wandb_utils import get_data_root_path, get_result_path


def main():
    args = setup_parser().parse_args()
    up_cen = args.up_cen 
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    log_dir = os.path.join(get_result_path(), param["project_name"])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("Result Dir: {}".format(log_dir))
    wandb.login(key="local-4e2ce713137197a03cbab2c3085d82a7518c1632", host="http://114.212.23.76:8080")
    wandb.init(project=param["project_name"], config=param, dir=log_dir,)
    param["model_name"] = "bofa"
    param["out_dir"] = wandb.run.dir
    if up_cen == 1:
        param["use_up_cen"] = True
    else:
        param["use_up_cen"] = False
    param["use_up_cov"] = True
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='exps/cifar_0_10.json',
                        help='Json file of settings.')
    parser.add_argument('--sample_num', type=int, default=0, )
    parser.add_argument('--Kt', type=int, default=256, help="Random seed.")
    parser.add_argument('--epoch', type=int, default=2, help="Random seed.")
    parser.add_argument("--project_name", type=str, default="test",
                        help="Project name of wandb")
    parser.add_argument("--img_only", action="store_true", default=False)
    parser.add_argument("--loss_type", type=str, default="CE", help="Loss type.")
    parser.add_argument("--center_type", type=str, default="mix", choices=["img", "text", "mix"])
    parser.add_argument("--up_cen", type=int, default=1)
    parser.add_argument("--save_feat", type=int, default=0)
    return parser


if __name__ == '__main__':
    main()
