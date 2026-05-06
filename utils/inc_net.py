import copy
import logging
import torch
from torch import nn
import timm
import torch.nn.functional as F
import os
import random


random.seed(1993)


def get_convnet(args, pretrained=False):
    backbone_name = args["convnet_type"].lower()
    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32', cache_dir="cache_dir")
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name == 'clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', cache_dir="cache_dir")
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name == 'openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', cache_dir="cache_dir")
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        return out

    def encode_text(self, x):
        text_features = self.model.encode_text(x)
        return text_features

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class BofaAdapter(BaseNet):
    def __init__(self, args, pretrained=None):
        super(BaseNet, self).__init__()

        self.model, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.visual = self.model.visual # vit
        self.visual_proj = self.visual.proj # qiaojieceng
        self.args = args
        self.freeze(self.model)  # 冻结主干网络

        # 任务相关状态
        self.task_id = 0
        self.label2task = {}

        # 类别统计信息
        self.mu = None
        self.mu_norm = None
        self.cov_inv = None
        self.cov_list = []
        self.update_cov = None
        self.current_W = None
        self.current_b = None

        self.original_visual_proj = self.visual_proj
        self.original_visual_proj.requires_grad = False  # 确保它不参与训练

        W0 = self.original_visual_proj.data # 预训练的权重
        in_dim, out_dim = W0.shape[0], W0.shape[1]  # 768 512
        self.hidden_dim_t = args["Kt"] # 256  安全正交子空间
        self.subspace_policy = args.get("subspace_policy", "data_oss") # add-4.27

        from convs.linears import OLF as OLF
        # A:[768,256] B[256,512]
        # self.olf_layer = OLF(in_features=in_dim, out_features=out_dim, W0_torch=W0.T, rank=self.hidden_dim_t)
        ####################add-4.27-start######################
        self.olf_layer = OLF(
            in_features=in_dim,
            out_features=out_dim,
            W0_torch=W0.T,
            rank=self.hidden_dim_t,
            subspace_policy=self.subspace_policy,
            basis_seed=args.get("basis_seed", 1993),
            basis_alloc=args.get("basis_alloc", "disjoint_block"),
            basis_eps=args.get("basis_eps", 1e-4),
            basis_zero_fix=args.get("basis_zero_fix", "near_zero_only"),
            ####################add-4.28-start######################
            shared_rank=args.get("shared_rank", -1),
            shared_lr_scale=args.get("shared_lr_scale", 0.1),
            ####################add-4.28-end########################
            ####################add-5.5-start######################
            shared_importance_mode=args.get("shared_importance_mode", "none"), # 'column_grad_scale'
            importance_beta=args.get("importance_beta", 0.9), # 0.9
            importance_alpha=args.get("importance_alpha", 1.0), # 1.0
            ####################add-5.5-end######################
            first_task_rank=args.get("first_task_rank", -1),
        )
        ####################add-4.27-end########################
        self.use_up_cov = args["use_up_cov"]
        self.classifier_list = nn.ModuleList()

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def visual_forward_(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self._expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)

        x = self.visual.ln_post(x)
        pooled, _ = self.visual._global_pool(x)
        return pooled

    @property
    def feature_dim(self):
        return self.model.out_dim

    def extract_vector(self, x):
        return self.model.encode_image(x)

    def update_task(self, cls_num):
        new_classifier = nn.Linear(768, cls_num)
        new_classifier.weight.data = self.current_W.T
        new_classifier.bias.data = self.current_b
        new_classifier.weight.requires_grad = True
        new_classifier.bias.requires_grad = True
        self.classifier_list.append(new_classifier)

    def start_train(self, cls_num):
        self.update_task(cls_num=cls_num)
        self.olf_layer.prepare_for_new_task()

    def prepare_stage2(self):
        self.olf_layer.prepare_for_stage2()

    def end_train(self):
        self.olf_layer.end_task()

    def encode_image(self, x, stage2=False, return_origin=False):
        input_features = self.visual_forward_(x) # [bs,768]
        norm_input_features = input_features / input_features.norm(dim=-1, keepdim=True)

        cls_results = []
        for cls in self.classifier_list:
            cls_results.append(cls(norm_input_features)) # 2ge [128,10]
        # x @ weight.T   [bs,768]@[768,512]=[bs,512]
        aligned_features = self.olf_layer(input_features, stage2=stage2)  # (batch, 512)

        if return_origin:
            origin_feature = input_features @ self.original_visual_proj
            return aligned_features, cls_results, origin_feature
        else:
            return aligned_features, cls_results
    
    def encode_image_eval(self, x):
        input_features = self.visual_forward_(x)
        norm_input_features = input_features / input_features.norm(dim=-1, keepdim=True)

        cls_results = []
        for cls in self.classifier_list:
            cls_results.append(cls(norm_input_features))

        aligned_features = self.olf_layer.eval_forward(input_features)  # (batch, 512)

        return aligned_features, cls_results

    """
    计算类别统计信息

    输入:
        known_classes: 已知类别数 (例如: 0, 10, 20...)
        total_classes: 总类别数 (例如: 10, 20, 30...)
        train_loader: 训练数据加载器

    输出:
        mu: 类别中心 [C, 768]  # C 是当前任务的类别数
        mu_norm: 归一化的类别中心 [C, 768]
        cov: 协方差矩阵 [768, 768]
        W, b: GDA 分类器参数
    """
    ############################Step 2: 统计信息更新######################################
    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("updating stat")
        with torch.no_grad():
            # 1. 提取所有训练样本的特征
            vecs = []  # 存储原始特征
            vecs_norm = []  # 存储归一化特征
            labels = []  # 存储标签
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device) # [bs=128,3,224,224]
                image_features = self.visual_forward_(inputs) # [bs,768]
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                vecs.append(image_features) # 图像特征
                vecs_norm.append(image_features_norm)
                labels.append(targets)

        for i in range(known_classes, total_classes):
            self.label2task[i] = self.task_id

        vecs = torch.cat(vecs) # [5000,768]
        #self.olf_layer.update_old_features(vecs) # 更新均值和协方差
        ####################add-4.27-start######################
        if self.olf_layer.uses_data_oss():
            self.olf_layer.update_old_features(vecs)
        ####################add-4.27-end########################
        vecs_norm = torch.cat(vecs_norm) # [5000,768]
        labels = torch.cat(labels) # [5000]
        # 2. 计算每个类别的中心  | 去中心化 = 减去均值 = 消除位置影响 ✅
        mu = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)], dim=0)
        center_vecs = torch.cat([vecs[labels == i] - mu[i - known_classes] for i in range(known_classes, total_classes)], dim=0)
        cov = torch.cov(center_vecs.t()) + 1e-4 * torch.eye(center_vecs.shape[-1]).to(device) # 协方差 = 衡量"共同变化",必须先去掉"位置"（均值），才能看到"变化"
        # [10,768]
        mu_norm = torch.cat([vecs_norm[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)], dim=0)
        center_vecs_norm = torch.cat([vecs_norm[labels == i] - mu_norm[i - known_classes]
                                     for i in range(known_classes, total_classes)], dim=0) # [5000,768]
        # 3. 计算协方差矩阵
        # 将所有样本减去各自类别中心
        cov_inv = center_vecs_norm.shape[1] * torch.linalg.pinv( # 计算协方差矩阵的逆=n * pinv((n-1) * Σ + trace(Σ) * I) | 求伪逆   (n-1) * Σ: 缩放因子+trace(Σ) * I: 对角线正则化（防止奇异）
            (center_vecs_norm.shape[0] - 1) * center_vecs_norm.T.cov() + center_vecs_norm.T.cov().trace() * torch.eye(center_vecs_norm.shape[1]).cuda())
        current_ps = torch.ones(mu_norm.shape[0]).cuda() * 1. / mu_norm.shape[0] # 计算先验概率 假设 每个类别的先验概率相等（均匀分布）
        self.current_W = torch.einsum('nd, dc -> cn', mu_norm, cov_inv) # 计算权重矩阵 W [10,768] [768,768]=[768,10]
        self.current_b = current_ps.log() - torch.einsum('nd, dc, nc -> n', mu_norm, cov_inv, mu_norm) / 2 # 计算偏置项 b 先验概率的对数 - 马氏距离的yiban μ_c.T @ Σ^(-1) @ μ_c

        if self.mu is None:
            self.mu = mu # [10,768]
            self.mu_norm = mu_norm
            self.cov_inv = cov_inv # [768,768] nijuzhen
            self.cov_list = [cov]
            self.update_cov = cov # [768,768]
        else:
            self.cov_inv = (known_classes / total_classes) * self.cov_inv + (total_classes - known_classes) / total_classes * cov_inv + (
                (known_classes / total_classes) * (total_classes - known_classes) / total_classes ** 2) * (
                self.mu_norm.T.mean(dim=1).unsqueeze(1) - mu_norm.T.mean(dim=1).unsqueeze(1)) @ (
                self.mu_norm.T.mean(dim=1).unsqueeze(1) - mu_norm.T.mean(dim=1).unsqueeze(1)).T
            self.update_cov = (known_classes / total_classes) * self.update_cov + (total_classes - known_classes) / total_classes * cov + (
                (known_classes / total_classes) * (total_classes - known_classes) / total_classes ** 2) * (
                self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)) @ (
                self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)).T
            self.mu = torch.cat([self.mu, mu])
            self.mu_norm = torch.cat([self.mu_norm, mu_norm])
            self.cov_list.append(cov)
        # 4. 计算 GDA (Gaussian Discriminant Analysis) 分类器参数
        # W: [C, 768] - GDA 分类器权重, b: [C] - GDA 分类器偏置
        # 用于后续的高斯判别分析分类 | GDA 假设每个类别的特征服从高斯分布，通过贝叶斯公式推导出的分类器可以写成线性形式 | GDA 分类器 → 转换为 → 线性分类器 (y = Wx + b)
        ps = torch.ones(self.mu_norm.shape[0]).cuda() * 1. / self.mu_norm.shape[0] # 计算先验概率 ps = [0.05, 0.05, ..., 0.05]  # 20个0.05
        self.W = torch.einsum('nd, dc -> cn', self.mu_norm, self.cov_inv)
        self.b = ps.log() - torch.einsum('nd, dc, nc -> n', self.mu_norm, self.cov_inv, self.mu_norm) / 2
        self.task_id += 1

    def sample_augmented_cls(self, classes: list, n: int):
        aug_features = []
        aug_labels = []

        for c in classes:
            if c not in self.label2task:
                raise ValueError(f"Class {c} not found in stored tasks")
            task_id = self.label2task[c]

            if self.use_up_cov:
                cov = self.update_cov
            else:
                cov = self.cov_list[task_id]

            mean = self.mu[c]
            vec = torch.randn(n, mean.shape[-1]).to(mean.device)
            sqrt_cov = torch.linalg.cholesky(cov)
            aug_c = vec @ sqrt_cov + mean

            aug_features.append(aug_c)
            aug_labels.extend([c] * n)

        X_aug = torch.cat(aug_features, dim=0)
        y_aug = torch.tensor(aug_labels, dtype=torch.long)
        return X_aug, y_aug

    def get_cls_center(self):
        return self.mu @ self.visual_proj

    def get_cls_center_last(self):
        with torch.no_grad():
            return self.olf_layer(self.mu) # [10,768]

    def get_cls_center_lora(self):
        with torch.no_grad():
            training_state = self.olf_layer.training # False
            self.olf_layer.eval() # self.mu:        [20, 768]  # 20个类别在原始特征空间的中心
            new_center = self.olf_layer(self.mu) # [20,512] 将原始特征空间（768维）的类别中心投影到 OLF Layer 输出空间（512维）
            self.olf_layer.train(training_state)
        return new_center # 20个类别在输出空间的中心

    def get_param_group(self):
        param_groups = []
        #param_groups.append({'params': self.olf_layer.get_trainable_parameters()})
        #param_groups.append({'params': self.olf_layer.get_stage2_parameters(), 'lr': 0.001, 'weight_decay': 0.001})

        ####################add-4.28-start######################
        trainable_params = self.olf_layer.get_trainable_parameters()
        # stage2_params = self.olf_layer.get_stage2_parameters()
        # if len(trainable_params) > 0:
        #     param_groups.append({'params': trainable_params})
        # if len(stage2_params) > 0:
        #     param_groups.append({'params': stage2_params, 'lr': 0.001, 'weight_decay': 0.001})
        if self.olf_layer._uses_shared_core():
            shared_params = self.olf_layer.get_shared_parameters() # [512,32]
            private_params = self.olf_layer.get_private_parameters() # [512,32]
            if len(shared_params) > 0:
                param_groups.append({
                    'params': shared_params, # # 共享参数：使用缩放后的学习率
                    'lr': self.args["init_lr"] * self.args.get("shared_lr_scale", 0.1), # 例如：0.01 * 0.1 = 0.001
                })
            if len(private_params) > 0: # 私有参数：使用正常学习率
                param_groups.append({'params': private_params})  # 使用默认的 init_lr，例如 0.01
        else:
            trainable_params = self.olf_layer.get_trainable_parameters()
            stage2_params = self.olf_layer.get_stage2_parameters()
            if len(trainable_params) > 0:
                param_groups.append({'params': trainable_params})
            if len(stage2_params) > 0:
                param_groups.append({'params': stage2_params, 'lr': 0.001, 'weight_decay': 0.001})

        ####################add-4.28-end########################

        def uses_shared_core(self):
            return self.olf_layer.uses_shared_core()

        if len(self.classifier_list) > 0:
            param_groups.append({'params': self.classifier_list[-1].parameters(),
                                'lr': 0.001, 'weight_decay': 0.001})

        return param_groups

    def train_state(self):
        self.olf_layer.train()

    def eval_state(self):
        self.olf_layer.eval()

    ####################add-4.27-start######################
    def set_total_tasks(self, total_tasks):
        self.olf_layer.set_total_tasks(total_tasks)

    def uses_two_stage(self):
        return self.olf_layer.uses_two_stage()
    ####################add-4.27-end########################

    ####################add-5.5-start######################
    def apply_shared_importance_to_grads(self):
        self.olf_layer.apply_shared_importance_to_grads()
    ####################add-5.5-end######################
