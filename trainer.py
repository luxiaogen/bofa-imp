import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np
import json
# import wandb
from datetime import datetime

####################add-5.12-start#######################
def _compact_float(value):
    return "{:g}".format(float(value)).replace(".", "p")


def _build_run_tag(args, timestamp):
    parts = [timestamp, "seed{}".format(args["seed"])]
    policy = args.get("subspace_policy", "data_oss")
    policy_alias = {
        "data_oss": "oss",
        "fixed_svd_basis": "svd",
        "fixed_svd_shared_core": "shared",
    }.get(policy, policy)
    parts.append(policy_alias)

    if policy != "data_oss":
        parts.append("Kt{}".format(args.get("Kt", 64)))
    if policy == "fixed_svd_shared_core":
        parts.append("sr{}".format(args.get("shared_rank", -1)))
    if args.get("epoch") is not None:
        parts.append("ep{}".format(args["epoch"]))
    if args.get("center_type"):
        parts.append("ct{}".format(args["center_type"]))
    if args.get("text_relation_lambda", 0.0) > 0:
        parts.append("rel{}".format(_compact_float(args["text_relation_lambda"])))
    if args.get("shared_svd_reg_lambda", 0.0) > 0:
        parts.append("svdl{}".format(_compact_float(args["shared_svd_reg_lambda"])))
        parts.append("top{}".format(args.get("shared_svd_reg_topk", 20)))
    if args.get("shared_svd_grad_mode", "none") != "none":
        parts.append(args["shared_svd_grad_mode"])
        parts.append("top{}".format(args.get("shared_svd_reg_topk", 20)))
    if args.get("shared_param_reg_mode", "none") != "none":
        parts.append(args["shared_param_reg_mode"])
        parts.append("pl{}".format(_compact_float(args.get("shared_param_reg_lambda", 0.0))))
    if args.get("shared_importance_mode", "none") != "none":
        parts.append(args["shared_importance_mode"])
        parts.append("ia{}".format(_compact_float(args.get("importance_alpha", 1.0))))
    return "_".join(parts)
####################add-5.12-end#######################

##########add.5.4-vis-start######################
def _compute_tensor_metrics(tensor):
    return {
        "fro_norm": float(torch.norm(tensor, p="fro").item()),
        "mean_abs": float(tensor.abs().mean().item()),
        "max_abs": float(tensor.abs().max().item()),
        "col_norms": [float(v) for v in torch.norm(tensor, dim=0).tolist()],
        "delta_from_zero": float(torch.norm(tensor, p="fro").item()),
    }


def _compute_b_summary(snapshot, previous_snapshot=None, eps=1e-8):
    summary = {
        "task_id": int(snapshot["task_id"]),
        "subspace_policy": snapshot["subspace_policy"],
        "basis_alloc": snapshot["basis_alloc"],
        "active_rank": int(snapshot["active_rank"]),
        "active_slice": list(snapshot["active_slice"]) if snapshot["active_slice"] is not None else None,
    }

    if "B" in snapshot:
        B = snapshot["B"]
        summary.update(_compute_tensor_metrics(B))
        # if previous_snapshot is not None and "B" in previous_snapshot:
        if (
                previous_snapshot is not None
                and "B" in previous_snapshot
                and tuple(previous_snapshot["B"].shape) == tuple(B.shape)
                and previous_snapshot.get("active_slice") == snapshot.get("active_slice")
        ):
            # summary["delta_fro_norm"] = float(torch.norm(B - previous_snapshot["B"], p="fro").item())
            summary["delta_reference"] = "previous_task"
        else:
            # summary["delta_fro_norm"] = summary["fro_norm"]
            summary["delta_reference"] = "zero_init"
    else:
        B_shared = snapshot["B_shared"]
        B_private = snapshot["B_private"]
        shared_metrics = _compute_tensor_metrics(B_shared)
        private_metrics = _compute_tensor_metrics(B_private)

        summary.update(
            {
                "shared_rank": int(snapshot["shared_rank"]),
                "private_rank": int(snapshot["private_rank"]),
                "private_slice": list(snapshot["private_slice"]) if snapshot["private_slice"] is not None else None,
                "fro_norm_shared": shared_metrics["fro_norm"],
                "fro_norm_private": private_metrics["fro_norm"],
                "mean_abs_shared": shared_metrics["mean_abs"],
                "mean_abs_private": private_metrics["mean_abs"],
                "max_abs_shared": shared_metrics["max_abs"],
                "max_abs_private": private_metrics["max_abs"],
                "col_norms_shared": shared_metrics["col_norms"],
                "col_norms_private": private_metrics["col_norms"],
                "delta_from_zero_shared": shared_metrics["delta_from_zero"],
                "delta_from_zero_private": private_metrics["delta_from_zero"],
                "shared_private_ratio": shared_metrics["fro_norm"] / (private_metrics["fro_norm"] + eps),
                "fro_norm": (shared_metrics["fro_norm"] ** 2 + private_metrics["fro_norm"] ** 2) ** 0.5,
            }
        )

        ####################add-5.5-start######################
        if "shared_importance" in snapshot:
            shared_importance = snapshot["shared_importance"]
            summary["shared_importance"] = [float(v) for v in shared_importance.tolist()]
            summary["shared_importance_mean"] = float(
                shared_importance.mean().item()) if shared_importance.numel() > 0 else 0.0
            summary["shared_importance_max"] = float(
                shared_importance.max().item()) if shared_importance.numel() > 0 else 0.0
        if "shared_grad_scale" in snapshot:
            shared_grad_scale = snapshot["shared_grad_scale"]
            summary["shared_grad_scale"] = [float(v) for v in shared_grad_scale.tolist()]
            summary["shared_grad_scale_mean"] = float(
                shared_grad_scale.mean().item()) if shared_grad_scale.numel() > 0 else 0.0
            summary["shared_grad_scale_min"] = float(
                shared_grad_scale.min().item()) if shared_grad_scale.numel() > 0 else 0.0
        if "shared_importance_mode" in snapshot:
            summary["shared_importance_mode"] = snapshot["shared_importance_mode"]
        ####################add-5.5-end######################
        ####################add-5.7-start######################
        if "shared_svd_reg_lambda" in snapshot:
            summary["shared_svd_reg_lambda"] = float(snapshot["shared_svd_reg_lambda"])
            summary["shared_svd_reg_topk"] = int(snapshot["shared_svd_reg_topk"])
            summary["shared_svd_grad_mode"] = snapshot.get("shared_svd_grad_mode", "none")
            summary["shared_svd_ready"] = bool(snapshot["shared_svd_ready"])
        if "shared_svd_weight" in snapshot:
            shared_svd_weight = snapshot["shared_svd_weight"]
            summary["shared_svd_weight"] = [float(v) for v in shared_svd_weight.tolist()]
            summary["shared_svd_weight_mean"] = float(
                shared_svd_weight.mean().item()) if shared_svd_weight.numel() > 0 else 0.0
            summary["shared_svd_weight_max"] = float(
                shared_svd_weight.max().item()) if shared_svd_weight.numel() > 0 else 0.0
        ####################add-5.7-end#########################
        ####################add-5.8-start######################
        if "shared_param_reg_mode" in snapshot:
            summary["shared_param_reg_mode"] = snapshot["shared_param_reg_mode"]
            summary["shared_param_reg_lambda"] = float(snapshot["shared_param_reg_lambda"])
            summary["shared_param_ready"] = bool(snapshot["shared_param_ready"])
        if "shared_param_importance" in snapshot:
            shared_param_importance = snapshot["shared_param_importance"]
            # summary["shared_param_importance"] = [float(v) for v in shared_param_importance.flatten().tolist()]
            summary["shared_param_importance_mean"] = (
                float(shared_param_importance.mean().item()) if shared_param_importance.numel() > 0 else 0.0
            )
            summary["shared_param_importance_max"] = (
                float(shared_param_importance.max().item()) if shared_param_importance.numel() > 0 else 0.0
            )
        ####################add-5.8-end#########################

        # if previous_snapshot is not None and "B_shared" in previous_snapshot and "B_private" in previous_snapshot:
        if (
                previous_snapshot is not None
                and "B_shared" in previous_snapshot
                and "B_private" in previous_snapshot
                and tuple(previous_snapshot["B_shared"].shape) == tuple(B_shared.shape)
                and tuple(previous_snapshot["B_private"].shape) == tuple(B_private.shape)
        ):
            delta_shared = float(torch.norm(B_shared - previous_snapshot["B_shared"], p="fro").item())
            delta_private = float(torch.norm(B_private - previous_snapshot["B_private"], p="fro").item())
            summary["delta_reference"] = "previous_task"
        else:
            delta_shared = shared_metrics["fro_norm"]
            delta_private = private_metrics["fro_norm"]
            summary["delta_reference"] = "zero_init"
        summary["delta_fro_norm_shared"] = delta_shared
        summary["delta_fro_norm_private"] = delta_private
        summary["delta_fro_norm"] = (delta_shared ** 2 + delta_private ** 2) ** 0.5

    return summary
######5.7-add-log-start##############################
# def _build_run_tag(args, timestamp):
#     parts = [
#         f"seed{args['seed']}",
#         args.get("subspace_policy", "data_oss"),
#         args.get("basis_alloc", "none"),
#         f"Kt{args.get('Kt', 'na')}",
#     ] # ['seed1993', 'fixed_svd_shared_core', 'shared_core_private_block', 'Kt498']
#
#     if args.get("shared_rank", -1) > 0:
#         parts.append(f"sr{args['shared_rank']}")
#
#     if args.get("shared_svd_grad_mode", "none") != "none":
#         parts.append(f"ogd{args.get('shared_svd_reg_topk', 20)}")
#
#     if args.get("shared_svd_reg_lambda", 0.0) > 0:
#         parts.append(f"svd{args.get('shared_svd_reg_topk', 20)}")
#         parts.append(f"lam{args['shared_svd_reg_lambda']}")
#
#     if args.get("shared_importance_mode", "none") != "none":
#         parts.append(f"impA{args.get('importance_alpha', 1.0)}B{args.get('importance_beta', 0.9)}")
#
#     parts.append(f"ep{args.get('tuned_epoch', 'na')}")
#     parts.append(timestamp)
#     return "_".join(str(p).replace("/", "-") for p in parts)
######5.7-add-log-end##############################
def _save_b_snapshot(model, analysis_dir, task_id, previous_snapshot=None):
    snapshot = model._network.olf_layer.get_b_snapshot()
    os.makedirs(analysis_dir, exist_ok=True)

    snapshot_path = os.path.join(analysis_dir, f"task_{task_id:02d}_b_snapshot.pt")
    torch.save(snapshot, snapshot_path)

    summary = _compute_b_summary(snapshot, previous_snapshot=previous_snapshot)
    summary_path = os.path.join(analysis_dir, f"task_{task_id:02d}_b_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Saved B snapshot to %s", snapshot_path)
    logging.info("B summary: %s", summary)
    return snapshot, summary
##########add.5.4-vis-end######################
def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    # 添加时间戳，格式: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # logfilename = "logs/{}/{}/{}/{}/seed:{}_model:{}_{}".format(args["model_name"],
    #                                                          args["dataset"],
    #                                                          init_cls, args["increment"],
    #                                                          args["seed"],
    #                                                          args["model_name"],
    #                                                          timestamp
    #                                                          )

    # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s",
    #                     handlers=[
    #                         logging.FileHandler(filename=logfilename + ".log"),
    #                         logging.StreamHandler(sys.stdout),
    #                     ],
    #                     )
    ##########################mod-5.4-vis-start####################################
    # run_tag = "seed_{}_model_{}_{}".format(args["seed"], args["model_name"], timestamp)
    run_tag = _build_run_tag(args, timestamp) # 'seed1993_fixed_svd_shared_core_shared_core_private_block_Kt498_sr468_ogd20_ep17_20260507_163947'
    logfilename = os.path.join(logs_name, run_tag + ".log")
    run_dir = os.path.join(logs_name, run_tag)
    analysis_dir = os.path.join(run_dir, "b_analysis")
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s",
                        handlers=[
                            logging.FileHandler(filename=logfilename),
                            logging.StreamHandler(sys.stdout),
                        ],
                        )
    ##########################mod-5.4-vis-end####################################

    _set_random()
    _set_device(args)
    print_args(args)
    ################################Step 1: 数据准备######################################
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"], )
    model = factory.get_model(args["model_name"], args)
    # model.save_dir = logs_name
    model.save_dir = run_dir # mod-5.4

    # 使用 W_fusion 权重（纯任务平均）
    top1_curve = {"top1": []}  # 基于 OLF Layer 的特征
    # 使用 W_fusion2 权重（任务 + 原始 CLIP）
    top2_curve = {"top1": []} # 基于 OLF Layer 的特征
    # 使用 GDA 分类器（高斯判别分析）
    top1_gda_curve = {"top1": []} # 结合 W_fusion 的集成预测
    # 使用 GDA 分类器
    top2_gda_curve = {"top1": []} # 结合 W_fusion2 的集成预测
    # 使用直接的 argmax 预测
    #max_curve = {"top1": []} #  基于 W_fusion
    #max2_curve = {"top1": []} # 基于 W_fusion2

    # task acc  cnn_accy:模型训练后在各个任务上的分类准确率统计
    top1_task_curve = {"top1": []}
    top2_task_curve = {"top1": []}
    top1_gda_task_curve = {"top1": []}
    top2_gda_task_curve = {"top1": []}

    previous_b_snapshot = None # add-5.4

    for task in range(data_manager.nb_tasks): # 10 task
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(
                count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        # cnn_accy, nme_accy = model.eval_task()
        cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = model.eval_task()
        ##########################add-5.4-vis-start####################################
        previous_b_snapshot, _ = _save_b_snapshot(model, analysis_dir, task, previous_snapshot=previous_b_snapshot)
        ##########################add-5.4-vis-end####################################

        model.after_task()

        ####################add-5.13-start#########################
        eval_items = [
            ("CNN-Wfusion", cnn_accy[0], top1_curve, top1_task_curve),
            ("CNN-Wfusion2", cnn_accy[1], top2_curve, top2_task_curve),
            ("GDA-Wfusion", cnn_accy[2], top1_gda_curve, top1_gda_task_curve),
            ("GDA-Wfusion2", cnn_accy[3], top2_gda_curve, top2_gda_task_curve),
        ]

        for name, acc, class_curve, task_curve in eval_items:
            logging.info("[Task Acc | %s] grouped: %s", name, acc["task_grouped"])
            task_curve["top1"].append(acc["task_top1"])
            logging.info("[Task Acc | %s] curve: %s", name, task_curve["top1"])
            logging.info(
                "[Task Acc | %s] average: %.4f",
                name,
                sum(task_curve["top1"]) / len(task_curve["top1"]),
            )

            logging.info("[Class Acc | %s] grouped: %s", name, acc["grouped"])
            class_curve["top1"].append(acc["top1"])
            logging.info("[Class Acc | %s] curve: %s", name, class_curve["top1"])
            logging.info(
                "[Class Acc | %s] average: %.4f",
                name,
                sum(class_curve["top1"]) / len(class_curve["top1"]),
            )
        ####################add-5.13-end###########################



        # logging.info("Task Top1: {}".format(cnn_accy[0]["task_grouped"]))
        # top1_task_curve["top1"].append(cnn_accy[0]["task_top1"])
        # logging.info("CNN task top1 curve: {}".format(top1_task_curve["top1"]))
        # logging.info("Average Task Accuracy (CNN): {}".format(sum(top1_task_curve["top1"]) / len(top1_task_curve["top1"])))
        #
        # logging.info("Task Top2: {}".format(cnn_accy[1]["task_grouped"]))
        # top2_task_curve["top1"].append(cnn_accy[1]["task_top1"])
        # logging.info("CNN task top2 curve: {}".format(top2_task_curve["top1"]))
        # logging.info("Average Task Accuracy (CNN Top2): {}".format(sum(top2_task_curve["top1"]) / len(top2_task_curve["top1"])))
        #
        # logging.info("Task GDA Top1: {}".format(cnn_accy[2]["task_grouped"]))
        # top1_gda_task_curve["top1"].append(cnn_accy[2]["task_top1"])
        # logging.info("GDA task top1 curve: {}".format(top1_gda_task_curve["top1"]))
        # logging.info("Average Task Accuracy (GDA): {}".format(sum(top1_gda_task_curve["top1"]) / len(top1_gda_task_curve["top1"])))
        #
        # logging.info("Task GDA Top2: {}".format(cnn_accy[3]["task_grouped"]))
        # top2_gda_task_curve["top1"].append(cnn_accy[3]["task_top1"])
        # logging.info("GDA task top2 curve: {}".format(top2_gda_task_curve["top1"]))
        # logging.info("Average Task Accuracy (GDA Top2): {}".format(sum(top2_gda_task_curve["top1"]) / len(top2_gda_task_curve["top1"])))
        #
        # ##########################################################################
        # logging.info("Top1: {}".format(cnn_accy[0]["grouped"]))
        # top1_curve["top1"].append(cnn_accy[0]["top1"])
        # logging.info("CNN top1 curve: {}".format(top1_curve["top1"]))
        # logging.info("Average Accuracy (CNN): {}".format(sum(top1_curve["top1"]) / len(top1_curve["top1"])))
        #
        # logging.info("Top2: {}".format(cnn_accy[1]["grouped"]))
        # top2_curve["top1"].append(cnn_accy[1]["top1"])
        # logging.info("CNN top2 curve: {}".format(top2_curve["top1"]))
        # logging.info("Average Accuracy (CNN Top2): {}".format(sum(top2_curve["top1"]) / len(top2_curve["top1"])))
        #
        # logging.info("GDA Top1: {}".format(cnn_accy[2]["grouped"]))
        # top1_gda_curve["top1"].append(cnn_accy[2]["top1"])
        # logging.info("GDA top1 curve: {}".format(top1_gda_curve["top1"]))
        # logging.info("Average Accuracy (GDA): {}".format(sum(top1_gda_curve["top1"]) / len(top1_gda_curve["top1"])))
        #
        # logging.info("GDA Top2: {}".format(cnn_accy[3]["grouped"]))
        # top2_gda_curve["top1"].append(cnn_accy[3]["top1"])
        # logging.info("GDA top2 curve: {}".format(top2_gda_curve["top1"]))
        # logging.info("Average Accuracy (GDA Top2): {}".format(sum(top2_gda_curve["top1"]) / len(top2_gda_curve["top1"])))

        #logging.info("Max Top1: {}".format(cnn_accy[4]["grouped"]))
        #max_curve["top1"].append(cnn_accy[4]["top1"])
        #logging.info("Max top1 curve: {}".format(max_curve["top1"]))
        #logging.info("Average Accuracy (Max): {}".format(sum(max_curve["top1"]) / len(max_curve["top1"])))

        #logging.info("Max Top2: {}".format(cnn_accy[5]["grouped"]))
        #max2_curve["top1"].append(cnn_accy[5]["top1"])
        #logging.info("Max top2 curve: {}".format(max2_curve["top1"]))
        #logging.info("Average Accuracy (Max Top2): {}".format(sum(max2_curve["top1"]) / len(max2_curve["top1"])))

        # wandb.log({"top1": cnn_accy[0]["top1"],
        #            "top2": cnn_accy[1]["top1"],
        #            "gda_top1": cnn_accy[2]["top1"],
        #            "gda_top2": cnn_accy[3]["top1"],
        #            "top1_avg": sum(top1_curve["top1"]) / len(top1_curve["top1"]),
        #            "top2_avg": sum(top2_curve["top1"]) / len(top2_curve["top1"]),
        #            "gda_top1_avg": sum(top1_gda_curve["top1"]) / len(top1_gda_curve["top1"]),
        #            "gda_top2_avg": sum(top2_gda_curve["top1"]) / len(top2_gda_curve["top1"]),
        #            "max_top1": cnn_accy[4]["top1"],
        #            "max2_top1": cnn_accy[5]["top1"],
        #            "max_top1_avg": sum(max_curve["top1"]) / len(max_curve["top1"]),
        #            "max2_top1_avg": sum(max2_curve["top1"]) / len(max2_curve["top1"]),
        #            })


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1993)
    np.random.seed(1993)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
