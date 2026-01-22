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
import wandb


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

    logfilename = "logs/{}/{}/{}/{}/seed:{}_model:{}".format(args["model_name"],
                                                             args["dataset"],
                                                             init_cls, args["increment"],
                                                             args["seed"],
                                                             args["model_name"],
                                                             )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s",
                        handlers=[
                            logging.FileHandler(filename=logfilename + ".log"),
                            logging.StreamHandler(sys.stdout),
                        ],
                        )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"], )
    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    top1_curve = {"top1": []}
    top2_curve = {"top1": []}
    top1_gda_curve = {"top1": []}
    top2_gda_curve = {"top1": []}
    max_curve = {"top1": []}
    max2_curve = {"top1": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(
                count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        # cnn_accy, nme_accy = model.eval_task()
        cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = model.eval_task()
        model.after_task()

        logging.info("Top1: {}".format(cnn_accy[0]["grouped"]))
        top1_curve["top1"].append(cnn_accy[0]["top1"])
        logging.info("CNN top1 curve: {}".format(top1_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(top1_curve["top1"]) / len(top1_curve["top1"])))

        logging.info("Top2: {}".format(cnn_accy[1]["grouped"]))
        top2_curve["top1"].append(cnn_accy[1]["top1"])
        logging.info("CNN top2 curve: {}".format(top2_curve["top1"]))
        logging.info("Average Accuracy (CNN Top2): {}".format(sum(top2_curve["top1"]) / len(top2_curve["top1"])))

        logging.info("GDA Top1: {}".format(cnn_accy[2]["grouped"]))
        top1_gda_curve["top1"].append(cnn_accy[2]["top1"])
        logging.info("GDA top1 curve: {}".format(top1_gda_curve["top1"]))
        logging.info("Average Accuracy (GDA): {}".format(sum(top1_gda_curve["top1"]) / len(top1_gda_curve["top1"])))

        logging.info("GDA Top2: {}".format(cnn_accy[3]["grouped"]))
        top2_gda_curve["top1"].append(cnn_accy[3]["top1"])
        logging.info("GDA top2 curve: {}".format(top2_gda_curve["top1"]))
        logging.info("Average Accuracy (GDA Top2): {}".format(sum(top2_gda_curve["top1"]) / len(top2_gda_curve["top1"])))

        logging.info("Max Top1: {}".format(cnn_accy[4]["grouped"]))
        max_curve["top1"].append(cnn_accy[4]["top1"])
        logging.info("Max top1 curve: {}".format(max_curve["top1"]))
        logging.info("Average Accuracy (Max): {}".format(sum(max_curve["top1"]) / len(max_curve["top1"])))

        logging.info("Max Top2: {}".format(cnn_accy[5]["grouped"]))
        max2_curve["top1"].append(cnn_accy[5]["top1"])
        logging.info("Max top2 curve: {}".format(max2_curve["top1"]))
        logging.info("Average Accuracy (Max Top2): {}".format(sum(max2_curve["top1"]) / len(max2_curve["top1"])))

        wandb.log({"top1": cnn_accy[0]["top1"],
                   "top2": cnn_accy[1]["top1"],
                   "gda_top1": cnn_accy[2]["top1"],
                   "gda_top2": cnn_accy[3]["top1"],
                   "top1_avg": sum(top1_curve["top1"]) / len(top1_curve["top1"]),
                   "top2_avg": sum(top2_curve["top1"]) / len(top2_curve["top1"]),
                   "gda_top1_avg": sum(top1_gda_curve["top1"]) / len(top1_gda_curve["top1"]),
                   "gda_top2_avg": sum(top2_gda_curve["top1"]) / len(top2_gda_curve["top1"]),
                   "max_top1": cnn_accy[4]["top1"],
                   "max2_top1": cnn_accy[5]["top1"],
                   "max_top1_avg": sum(max_curve["top1"]) / len(max_curve["top1"]),
                   "max2_top1_avg": sum(max2_curve["top1"]) / len(max2_curve["top1"]),
                   })


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
