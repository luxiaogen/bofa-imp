import json
import argparse
from trainer import train
# import wandb
import os
# from utils.wandb_utils import get_data_root_path, get_result_path



def main():
    args = setup_parser().parse_args()
    up_cen = args.up_cen 
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    # log_dir = os.path.join(get_result_path(), param["project_name"])
    # 使用本地路径替代 wandb 路径
    log_dir = os.path.join("./logs", param["project_name"])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("Result Dir: {}".format(log_dir))
    # wandb.login(key="local-4e2ce713137197a03cbab2c3085d82a7518c1632", host="http://114.212.23.76:8080")
    # wandb.init(project=param["project_name"], config=param, dir=log_dir,)
    param["model_name"] = "bofa"
    # param["out_dir"] = wandb.run.dir
    param["out_dir"] = log_dir  # 使用本地路径
    if up_cen == 1: # Use Updated Center（使用更新的类别中心） 训练前统计类别中心 → 训练时每个batch更新中心 → 推理时使用优化后的中心
        param["use_up_cen"] = True # 控制在训练时，类别中心（class prototype）是固定的还是动态更新的 EMA update
    else:
        param["use_up_cen"] = False # 训练前统计类别中心 → 训练时使用固定中心 → 推理时使用固定中心
    param["use_up_cov"] = True
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/cifar_0_10.json',
                        help='Json file of settings.')
    parser.add_argument('--sample_num', type=int, default=0, )
    # parser.add_argument('--Kt', type=int, default=256, help="Random seed.")
    parser.add_argument('--epoch', type=int, default=2, help="Random seed.")
    parser.add_argument("--project_name", type=str, default="test",
                        help="Project name of wandb")
    parser.add_argument("--img_only", action="store_true", default=False)
    parser.add_argument("--loss_type", type=str, default="CE", help="Loss type.")
    parser.add_argument("--center_type", type=str, default="mix", choices=["img", "text", "mix"])
    parser.add_argument("--up_cen", type=int, default=1)
    parser.add_argument("--save_feat", type=int, default=0)

    parser.add_argument('--Kt', type=int, default=64, help="Per-task subspace rank.")

    parser.add_argument(
        "--subspace_policy",
        type=str,
        default="data_oss",
        # choices=["data_oss", "fixed_svd_basis", "fixed_fullrank_spectrum"],
        choices=["data_oss", "fixed_svd_basis", "fixed_svd_shared_core"],
        help="Subspace construction policy for OLF.",
    )
    parser.add_argument("--basis_seed", type=int, default=1993, help="Random seed for fixed basis generation.")
    parser.add_argument("--basis_alloc", type=str, default="disjoint_block", help="Task subspace allocation policy.")


    ####################add-4.28-start######################
    parser.add_argument("--shared_rank", type=int, default=-1, help="Shared-core rank for fixed_svd_shared_core.")
    parser.add_argument("--shared_lr_scale", type=float, default=0.1,help="Learning-rate scale for shared-core updates.")
    ####################add-4.28-end########################
    parser.add_argument("--first_task_rank", type=int, default=-1,
                        help="Override rank for task 0 when basis_alloc=front_loaded_block.")
    ####################add-5.5-start######################
    parser.add_argument(
        "--shared_importance_mode",
        type=str,
        default="none",
        choices=["none", "column_grad_scale"],
        help="Importance-aware update mode for shared-core B_shared.",
    )
    parser.add_argument("--importance_beta", type=float, default=0.9, help="EMA decay for shared importance.")
    parser.add_argument("--importance_alpha", type=float, default=1.0, help="Strength of shared gradient suppression.")
    ####################add-5.5-end######################
    ####################add-5.7-start######################
    parser.add_argument("--shared_svd_reg_lambda", type=float, default=0.0,
                        help="Regularization strength for preserving top-k SVD directions of B_shared.")
    parser.add_argument("--shared_svd_reg_topk", type=int, default=20,
                        help="Number of top singular directions of B_shared to preserve.")
    parser.add_argument("--shared_svd_grad_mode",type=str,default="none",choices=["none", "ogd_project"],help="Gradient projection mode for preserving top-k SVD directions of B_shared.")
    ####################add-5.7-end#########################
    ####################add-5.8-start######################
    parser.add_argument(
        "--shared_param_reg_mode",
        type=str,
        default="none",
        choices=["none", "ewc_grad"],
        help="Parameter-level regularization mode for B_shared.",
    )
    parser.add_argument("--shared_param_reg_lambda", type=float, default=0.0,
                        help="Regularization strength for EWC-style B_shared consolidation.")
    parser.add_argument("--shared_param_importance_beta", type=float, default=0.9,
                        help="EMA decay for EWC-style B_shared parameter importance.")
    ####################add-5.8-end#########################
    ####################add-5.12-start#########################
    # parser.add_argument(
    #     "--proto_select_mode",
    #     type=str,
    #     default="none",
    #     choices=["none", "topk_image_then_mix", "topk_pairwise_mix"],
    #     help="Optional text-guided top-k image prototype fusion mode for center_type=mix.",
    # )
    # parser.add_argument("--proto_select_topk", type=int, default=0,
    #                     help="Top-k image prototypes selected by each text prototype. 0 disables selection.")
    # parser.add_argument("--proto_select_tau", type=float, default=0.07,
    #                     help="Softmax temperature for text-guided top-k prototype selection.")
    ####################add-5.12-end###########################

    return parser


if __name__ == '__main__':
    main()
