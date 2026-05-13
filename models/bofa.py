import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from utils.inc_net import BofaAdapter
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute
import random
random.seed(1993)
np.random.seed(1993)

num_workers = 8
# Learner 是增量学习的总指挥！
# Learner 就是整个 BOFA 系统的"大脑":✅ 管理模型（BofaAdapter） ✅ 控制训练流程（Stage 1/2）✅ 管理任务状态（已学类别、当前任务） ✅ 存储和更新类别原型 ✅ 评估模型性能
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self._train_transformer = False
        self._network = BofaAdapter(args) # CLIP
        self._network.eval()

        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None) # visual
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5) # 15
        self.stage2_epoch = get_attribute(args, "epoch", 2) # 2
        self._known_classes = 0
        self.prototype = []
        self.loss_type = get_attribute(args, "loss_type", "CE")
        # last_mask
        self.last_mask = get_attribute(args, "last_mask", False) # False
        self.use_up_cen = get_attribute(args, "use_up_cen", False) # True 每个 batch 动态更新类别中心，使用指数移动平均 img_proto = 0.95 * img_proto + 0.05 * new_proto
        self.center_type = get_attribute(args, "center_type", "mix") # 混合图像+文本 控制使用哪种类别中心进行分类
        ####################add-5.12-start#########################
        # self.proto_select_mode = get_attribute(args, "proto_select_mode", "none") # topk_pairwise_mix
        # self.proto_select_topk = get_attribute(args, "proto_select_topk", 0)
        # self.proto_select_tau = get_attribute(args, "proto_select_tau", 0.07)
        ####################add-5.12-end###########################
        self.t_lam = 0
        self.stat = args['stat']
        self.label2task = []
        self.train_loader_list = []
        self.test_loader_list = []
        self.first_task = True

    def after_task(self):
        self._known_classes = self._total_classes

    def _get_batch_des(self, des_file, classnames):
        batch_des = []
        for classname in classnames:
            batch_des.append(classname + ' with ' + random.choice(des_file[classname]).lower())
        return batch_des

    ####################add-5.12-start#########################
    # def _normalize_proto(self, proto):
    #     return proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    #
    # def _build_mix_proto(self, img_proto, text_proto, lam=None):
    #     lam = self.t_lam if lam is None else lam # 0.55
    #     img_proto = self._normalize_proto(img_proto)
    #     text_proto = self._normalize_proto(text_proto)
    #
    #     if self.proto_select_mode == "none" or self.proto_select_topk <= 0:
    #         return lam * img_proto + (1 - lam) * text_proto # 原来的那一种混合prototype，不做调整
    #
    #     topk = min(int(self.proto_select_topk), img_proto.shape[0])
    #     tau = max(float(self.proto_select_tau), 1e-6) # 温度
    #     sim = text_proto @ img_proto.t() # [20,20] 第 i 个文本原型 和 第 j 个图像原型 有多像
    #     topk_values, topk_indices = torch.topk(sim, k=topk, dim=1) #
    #     topk_weight = torch.softmax(topk_values / tau, dim=1) # [C=20,topk=3]
    #     selected_img_proto = img_proto[topk_indices] # [C=20,topk=3,512]  语义邻居融合 图像与文本最相近的 topk 个图像原型
    #
    #     if self.proto_select_mode == "topk_image_then_mix":
    #         selective_img_proto = (topk_weight.unsqueeze(-1) * selected_img_proto).sum(dim=1) # [20,512]
    #         selective_img_proto = self._normalize_proto(selective_img_proto)
    #         return lam * selective_img_proto + (1 - lam) * text_proto
    #
    #     if self.proto_select_mode == "topk_pairwise_mix":
    #         text_proto_expand = text_proto.unsqueeze(1).expand(-1, topk, -1) # [20,512]->[20,topk=3,512]
    #         pairwise_proto = lam * selected_img_proto + (1 - lam) * text_proto_expand # [20,topk=3,512]
    #         pairwise_proto = self._normalize_proto(pairwise_proto)
    #         cls_proto = (topk_weight.unsqueeze(-1) * pairwise_proto).sum(dim=1) # [20,512]
    #         return self._normalize_proto(cls_proto)

        # raise ValueError("Unknown proto_select_mode: {}".format(self.proto_select_mode))

    # def _build_cls_proto(self, img_proto, text_proto):
    #     if self.center_type == "img":
    #         return self._normalize_proto(img_proto)
    #     if self.center_type == "text":
    #         return self._normalize_proto(text_proto)
    #     return self._build_mix_proto(img_proto, text_proto)
    ####################add-5.12-end###########################

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train", mode="train") # 5000
        test_dataset_task = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                     source="test", mode="test") # 1000
        train_eval_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="test") # 5000

        train_eval_loader = DataLoader(train_eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.train_loader_list.append(train_eval_loader) # every task trainloader
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
        cur_label2task = [self._cur_task] * (self._total_classes - self._known_classes)
        self.label2task = self.label2task + cur_label2task  # label2task[类别ID] = 任务ID
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader_task = DataLoader(test_dataset_task, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader_list.append(self.test_loader_task)
        if len(self._multiple_gpus) > 1: # multi gpu
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1: # multi gpu
            self._network = self._network.module
        ####################add-4.27-start######################
        self._network.set_total_tasks(data_manager.nb_tasks)
        ####################add-4.27-end########################
        self._network.update_stat(self._known_classes, self._total_classes, self.train_loader, self._device)
        self.init_accuracy(self.train_loader, self.test_loader_task, self.test_loader)
        # self._network.update_task(self._total_classes - self._known_classes)
        self._network.start_train(self._total_classes - self._known_classes)
        self.train(self.train_loader, self.test_loader, train_dataset)
        self._network.end_train()

    def eval_init(self, eval_loader, text_proto):
        text_correct, all_num = 0, 0
        for i, (_, inputs, targets) in enumerate(eval_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features, logits, origin_image_features = self._network.encode_image(inputs, return_origin=True)
                origin_image_features = origin_image_features / origin_image_features.norm(dim=-1, keepdim=True)
                text_outputs = (origin_image_features @ text_proto.T)
                text_pred = torch.max(text_outputs, dim=1)[1].cpu()
                text_correct += text_pred.eq(targets).cpu().sum()
                all_num += len(targets)
        return np.around(tensor2numpy(text_correct) * 100 / all_num, decimals=2)

    @torch.no_grad()
    def search_lambda_for_prompt(self, eval_loader, image_proto, text_proto, num_grid: int = 21):
        image_proto = image_proto / image_proto.norm(dim=-1, keepdim=True)  # [C, D]
        text_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)  # [C, D]

        all_feats, all_labels = [], []
        for _, imgs, labels in eval_loader:
            imgs = imgs.to(self._device)
            feats, _, _ = self._network.encode_image(imgs, return_origin=True)  # feats: [B, D]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
        all_feats = torch.cat(all_feats,  dim=0)   # [N=5000, D]
        all_labels = torch.cat(all_labels, dim=0)   # [N]

        best_acc, best_lam, best_proto = -1.0, 0.0, None
        lambdas = torch.linspace(0, 1, steps=num_grid)

        for lam in lambdas: # [0.0, 0.05, 0.1, ..., 1.0]:  # 21 个候选值
            # 融合原型
            new_proto = (1 - lam) * image_proto + lam * text_proto   # [C, D] # mod-5.12
            ####################add-5.12-start#########################
            # if self.proto_select_mode == "none" or self.proto_select_topk <= 0:
            #     new_proto = (1 - lam) * image_proto + lam * text_proto  # [C, D]
            # else:
            #     new_proto = self._build_mix_proto(image_proto, text_proto, lam=lam.item())  # [C, D]
            ####################add-5.12-end###########################
            logits = all_feats @ new_proto.T.cpu()   # [N, C]
            pred = logits.argmax(dim=1)

            acc = (pred == all_labels).float().mean().item()      # 0~1

            if acc > best_acc:
                best_acc = acc
                best_lam = lam.item()
                best_proto = new_proto.clone()

        print(f"\n>>> best λ = {best_lam:.3f}")
        return best_lam
    ################################Step 3: 初始化准确率评估######################################
    def init_accuracy(self, train_loader, test_new_loader, test_loader):
        """
            搜索最佳的图像-文本融合系数 λ

            流程:
            1. 生成文本原型 (text_proto)
            2. 获取图像原型 (image_proto)
            3. 搜索最佳 λ 使得融合原型分类准确率最高
        """
        # 1. 生成文本原型
        # 对每个类别，使用 CLIP 文本编码器编码类别名
        class_to_label = self.data_manager._class_to_label # 100
        templates = self.data_manager._data_to_prompt[0]
        labels = [class_to_label[y] for y in range(self.args['init_cls'] + self._cur_task * self.args['increment'])]
        texts = [templates.format(inst) for inst in labels]
        texts = self._network.tokenizer(texts).to(self._device) # [10,77]
        self.text_proto = self._network.encode_text(texts) # [10,512]

        # 2. 获取图像原型
        # 使用之前计算的类别中心投影到 CLIP 空间
        image_proto = self._network.get_cls_center() # [10,512]
        if self.t_lam == 0:
            # 3. 搜索最佳 λ
            self.t_lam = self.search_lambda_for_prompt(train_loader, image_proto, self.text_proto)
        new_proto = image_proto / image_proto.norm(dim=-1, keepdim=True) * (1 - self.t_lam) + \
           self.text_proto / self.text_proto.norm(dim=-1, keepdim=True) * self.t_lam # mod-5.12
        # ####################add-5.12-start#########################
        # if self.proto_select_mode == "none" or self.proto_select_topk <= 0:
        #     new_proto = image_proto / image_proto.norm(dim=-1, keepdim=True) * (1 - self.t_lam) + \
        #                 self.text_proto / self.text_proto.norm(dim=-1, keepdim=True) * self.t_lam
        # else:
        #     new_proto = self._build_cls_proto(image_proto, self.text_proto)
        ####################add-5.12-end###########################
        test_acc_lam = self.eval_init(test_loader, new_proto)
        logging.info("Eval Test Loader: Zero_Shot_Lam: {:.2f}".format(test_acc_lam))

    ################################Step 4: 训练循环 (train)######################################
    def train(self, train_loader, test_loader, train_dataset):
        """
            训练循环

            分为两个阶段:
            - Stage 1: 训练 OLF Layer 和分类器
            - Stage 2: 微调 (仅在非首任务时)
        """
        self._network.to(self._device)
        param_groups = self._network.get_param_group()
        lr = self.init_lr
        weight_decay = self.weight_decay
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params=param_groups, momentum=0.9, lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params=param_groups, lr=self.init_lr, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [4, 10], gamma=0.1, last_epoch=-1)

        class_to_label = self.data_manager._class_to_label # 100ge biao qian ming
        #prog_bar = tqdm(range(self.tuned_epoch+self.stage2_epoch)) # 15 + 2
        ####################add-4.27-start######################
        total_epochs = self.tuned_epoch + self.stage2_epoch if self._network.uses_two_stage() else self.tuned_epoch
        prog_bar = tqdm(range(total_epochs))
        ####################add-4.27-end########################
        templates = self.data_manager._data_to_prompt[0] # 'a good photo of a {}.'
        # if self._cur_task > 0:
        #     old_class = list(range(self.args['init_cls'] + (self._cur_task - 1) * self.args['increment']))
        from utils.toolkit import ClipLoss
        cliploss = ClipLoss(img_only=True)
        text_proto = self.text_proto # [20,512]
        img_proto = self._network.get_cls_center() # [20,512]
        for _, epoch in enumerate(prog_bar): # 17 epoch
            self._network.train_state()
            self._network.train()
            losses = 0.0
            loss_low = 0.0 # 低层分类损失 只在前6个epoch使用 让 OLF Layer 学会区分当前任务的类别
            loss_clip = 0.0

            correct, total = 0, 0
            # 判断是否进入 Stage 2
            #if epoch == self.tuned_epoch and self._cur_task > 0:
            if epoch == self.tuned_epoch and self._cur_task > 0 and self._network.uses_two_stage():
                self._network.prepare_stage2() # 激活 Stage 2 参数 B
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.use_up_cen:
                    new_proto = self._network.get_cls_center_last() # [20,512] 图像特征的均值mu经过桥接层
                    img_proto = 0.95 * img_proto + 0.05 * new_proto # [20,512]
                self.img_proto = img_proto
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                if self._cur_task > 0: # 局部标签 = 全局标签 - 当前任务的起始类别  zhuan huan dao 0-9
                    offset_targets = targets - self.args['init_cls'] - (self._cur_task - 1) * self.args['increment']
                else:
                    offset_targets = targets
                logit_scale = self._network.model.logit_scale # suo fang yin zi 4.6
                # 1. 前向传播
                #if epoch >= self.tuned_epoch and self._cur_task > 0:
                if epoch >= self.tuned_epoch and self._cur_task > 0 and self._network.uses_two_stage(): # add-4.27
                    image_features, low_logits = self._network.encode_image(inputs, stage2=True, return_origin=False)
                else: # 对齐后的图像特征:[bs,512], 分类头的输出 logits:[bs,20]
                    image_features, low_logits = self._network.encode_image(inputs, return_origin=False)
                low_logits = low_logits[-1]# [128,20] 当前任务的
                if epoch < 6: # L_low = CE(low_logits, 当前任务内标签)
                    low_loss = nn.functional.cross_entropy(low_logits, offset_targets) # 让 low_logits 这个分支也具备当前任务分类能力，属于辅助监督 cross_entropy：让真实类别对应的 logit 变大，其他类别的 logit 变小
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)
                if self.loss_type == "CE":
                    if self.center_type == "img":
                        cls_proto = img_proto / img_proto.norm(dim=-1, keepdim=True)
                    elif self.center_type == "text":
                        cls_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)
                    else:
                        cls_proto = self.t_lam * (img_proto / img_proto.norm(dim=-1, keepdim=True)) + \
                            (1 - self.t_lam) * text_proto / text_proto.norm(dim=-1, keepdim=True)
                    # cls_proto = self._build_cls_proto(img_proto, text_proto) # add-5.12
                    logits = self._network.model.logit_scale * img_feas @ cls_proto.t() # [128,512]@[10,512]
                    clip_loss = nn.functional.cross_entropy(logits, targets) # 交叉熵，图像特征和类别原型的相似度 logits
                else:
                    labels = [class_to_label[y] for y in targets]
                    texts_clip = [templates.format(inst) for inst in labels]
                    clip_text_feas = self._network.encode_text(self._network.tokenizer(texts_clip).to(self._device))
                    clip_text_norm = clip_text_feas.norm(dim=-1, keepdim=True)
                    clip_text_feas = clip_text_feas / clip_text_norm
                    clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)

                # 2.1 Low-level 损失 (前 6 个 epoch)
                if epoch < 6:
                    #
                    loss = low_loss + clip_loss
                else:
                    loss = clip_loss # 2.2 CLIP 损失
                # 3. 反向传播

                loss = loss + self._network.shared_svd_regularization() # add-5.7
                ####################add-5.8-start######################
                loss = loss + self._network.shared_param_regularization()
                ####################add-5.8-end#########################


                optimizer.zero_grad()
                loss.backward()

                ####################add-5.5-start######################
                self._network.apply_shared_svd_ogd_to_grads()
                self._network.apply_shared_importance_to_grads()
                ####################add-5.5-end######################

                ####################add-5.8-start######################
                self._network.accumulate_shared_param_importance()
                ####################add-5.8-end#########################

                optimizer.step()
                losses += loss.item()
                loss_low += low_loss.item()
                loss_clip += clip_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss_Clip {:.3f}, Train_acc {:.2f}".format(
                # self._cur_task, epoch + 1, self.args['tuned_epoch'], loss_clip / len(train_loader), train_acc)
                self._cur_task, epoch + 1, total_epochs, loss_clip / len(train_loader), train_acc)

            logging.info(info)
            prog_bar.set_description(info)

    def _compute_accuracy(self, model, loader, epoch=0):
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings.mean(dim=0)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        text_proto = text_features.to(self._device)

        img_proto = self.img_proto

        cls_proto = self.t_lam * (img_proto / img_proto.norm(dim=-1, keepdim=True)) + \
            (1 - self.t_lam) * text_proto / text_proto.norm(dim=-1, keepdim=True)

        ####################add-5.12-start#########################
        # cls_proto = self._build_cls_proto(img_proto, text_proto)
        ####################add-5.12-end###########################
        cls_proto2 = self._network.get_cls_center_lora()
        cls_proto2 = cls_proto2 / cls_proto2.norm(dim=-1, keepdim=True)
        correct, correct_2, total = 0, 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            if self._cur_task > 0:
                offset_targets = targets - self.args['init_cls'] - (self._cur_task - 1) * self.args['increment']
            else:
                offset_targets = targets
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features, logits = self._network.encode_image(inputs)
                logits = logits[-1]
                transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
                outputs = (transf_image_features @ cls_proto.T)

            predicts = torch.max(logits, dim=1)[1]
            predicts2 = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == offset_targets).sum()
            correct_2 += (predicts2.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2), np.around(tensor2numpy(correct_2) * 100 / total, decimals=2)

    def ens_result(self, logits_list, origin_predicts, outputs_img, outputs_gda):
        class_drift = [0]
        for i in range(len(logits_list)):
            class_drift.append(class_drift[i] + logits_list[i].shape[1])
        logits_sum = torch.cat(logits_list, dim=1)
        best_class_indices = [torch.argmax(logits_list[i], dim=1) + class_drift[i] for i in range(len(logits_list))]
        best_class_indices = torch.stack(best_class_indices, dim=1)
        selected_logits = torch.gather(origin_predicts, 1, best_class_indices)
        final_predicts = torch.argmax(selected_logits, dim=1)
        final_predicts = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts]

        selected_logits_img = torch.gather(outputs_img, 1, best_class_indices)
        final_predicts_img = torch.argmax(selected_logits_img, dim=1)
        final_predicts_img = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts_img]

        outputs_gda = self.stat * outputs_gda + (1 - self.stat) * outputs_img
        selected_logits_gda = torch.gather(outputs_gda, 1, best_class_indices)
        final_predicts_gda = torch.argmax(selected_logits_gda, dim=1)
        final_predicts_gda = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts_gda]
        return logits_sum, final_predicts, final_predicts_img, final_predicts_gda

    def ens_two_stage(self, best_class_indices, outputs):
        # best_class_indices: [bs,num_tasks] - 每个任务的候选类别
        # outputs: [bs,total_classes] - 所有类别的得分

        # 1. 从所有类别得分中，只取候选类别的得分
        selected_logits = torch.gather(outputs, 1, best_class_indices) # [B, num_tasks]
        # 2. 在候选类别中选最高得分的
        final_predicts = torch.argmax(selected_logits, dim=1) # [B]
        # 3. 转换回全局类别ID
        final_predicts = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts]
        return final_predicts
    # 两阶段集成预测：先在每个任务内选最优，再在任务间选最优 | logits_list: 每个任务分类器的输出,out_update: 基于类别原型的得分,out_gda: GDA 分类器的得分
    def get_ens_result(self, logits_list, out_update, out_gda):
        class_drift = [0]
        for i in range(len(logits_list)): # Step 1: 计算类别偏移量
            class_drift.append(class_drift[i] + logits_list[i].shape[1]) # [0, 10, 20]
        best_class_indices = [torch.argmax(logits_list[i], dim=1) + class_drift[i] for i in range(len(logits_list))] # Step 2: 找每个任务的最优类别(0->映射到对应的类别编号)
        best_class_indices = torch.stack(best_class_indices, dim=1)
        # Step 3: 融合 GDA 和原型得分
        out_ens_gda = self.stat * out_gda + (1 - self.stat) * out_update # GDA 分类器的得分,基于类别原型的得分
        out_ens_gda = self.ens_two_stage(best_class_indices, out_ens_gda) # out_ens: 集成预测（不用 GDA）
        out_ens = self.ens_two_stage(best_class_indices, out_update) # out_ens_gda: 集成预测（用 GDA）

        return out_ens, out_ens_gda
    #  W = mu @ cov_inv.T  # [768, C]   b = log(ps) - (1/2) * diag(mu @ cov_inv @ mu.T)  # [C]
    def gda_pred(self, inputs): # W 和 b 是在每个任务的 update_stat() 中，根据训练数据的统计信息（均值、协方差）计算得到的 GDA 分类器参数！ ✅
        transf_image_features_raw_ = self._network.visual_forward_(inputs)
        transf_image_features_raw_ = transf_image_features_raw_ / transf_image_features_raw_.norm(dim=-1, keepdim=True)
        outputs_gda = transf_image_features_raw_ @ self._network.W + self._network.b
        return outputs_gda

    def get_result(self, transf_image_features, cls_proto, inputs, logits):
        transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
        # 方式 1: 原型匹配
        out_update = (transf_image_features @ cls_proto.T) # Wfuse
        # 方式 2: GDA 预测
        out_gda = self.gda_pred(inputs) # [bs,20]
        out_pred, out_pred_gda = self.get_ens_result(logits, out_update, out_gda)
        out_argmax = torch.argmax(out_update, dim=1)
        return out_pred, out_pred_gda,out_argmax

    def _eval_cnn(self, loader):
        self._network.to(self._device)
        class_to_label = self.data_manager._class_to_label # label
        templates = self.data_manager._data_to_prompt # 18 ge mo ban
        total_labels = class_to_label[:self._total_classes]  # 所有已经学过的类别 mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0) # [所有已经学过的类别e.g.=20,512]
        text_proto = text_features.to(self._device) # [20,512]

        
        img_proto = self._network.get_cls_center_lora() # [20,512]
        if self.center_type == "img":
            cls_proto = img_proto / img_proto.norm(dim=-1, keepdim=True)
        elif self.center_type == "text":
            cls_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)
        else:
            if self.use_up_cen:
                cls_proto = self.t_lam * (img_proto / img_proto.norm(dim=-1, keepdim=True)) + \
                    (1 - self.t_lam) * text_proto / text_proto.norm(dim=-1, keepdim=True)
            else:
                img_proto2 = self._network.get_cls_center()
                cls_proto = self.t_lam * (img_proto2 / img_proto2.norm(dim=-1, keepdim=True)) + \
                    (1 - self.t_lam) * text_proto / text_proto.norm(dim=-1, keepdim=True)
        ####################add-5.12-start#########################
        # if self.center_type == "mix" and not self.use_up_cen:
        #     img_proto = self._network.get_cls_center()
        # cls_proto = self._build_cls_proto(img_proto, text_proto)
        ####################add-5.12-end###########################
        y_true = [] # 	真实标签
        y_pred = [] # Top1 预测 | W_fusion + 集成
        y_pred_gda = [] # 	Top2 预测 | W_fusion2 + 集成
        y_pred2 = [] # GDA Top1 | GDA + W_fusion
        y_pred2_gda = [] # GDA Top2 | GDA + W_fusion2
        y_max = [] # Max Top1 | W_fusion + argmax
        y_max2 = [] # 	Max Top2 | W_fusion2 + argmax
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features, logits = self._network.encode_image_eval(inputs) # list[len=两种融合权重的结果,[bs,512]]
                transf_image_features1, transf_image_features2 = transf_image_features # [bs,512]
                """
                    out_pred	类别原型 + 集成(用集成（两阶段）比如两个分类器选出两个结果，然后再两个结果中再找出一个正确的类别)
                    out_pred_gda 	GDA + 原型 + 集成
                    out_max      	类别原型 + argmax
                """
                out_pred, out_pred_gda,out_max = self.get_result(transf_image_features1, cls_proto, inputs, logits)
                # shiyong w_fuse2 de
                out_pred2, out_pred2_gda, out_max = self.get_result(transf_image_features2, cls_proto, inputs, logits)

            y_true.append(targets.cpu().numpy())
            y_pred.append(out_pred.cpu().numpy())
            y_pred2.append(out_pred2.cpu().numpy())
            y_pred_gda.append(out_pred_gda.cpu().numpy())
            y_pred2_gda.append(out_pred2_gda.cpu().numpy())
            y_max.append(out_max.cpu().numpy())
            y_max2.append(out_max.cpu().numpy())

        # [N, topk]
        return [np.concatenate(y_pred), np.concatenate(y_pred2), np.concatenate(y_pred_gda), np.concatenate(y_pred2_gda),np.concatenate(y_max), np.concatenate(y_max2)
                ], np.concatenate(y_true)
