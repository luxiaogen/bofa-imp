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
        self.visual = self.model.visual
        self.visual_proj = self.visual.proj
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

        W0 = self.original_visual_proj.data
        in_dim, out_dim = W0.shape[0], W0.shape[1]
        self.hidden_dim_t = args["Kt"]

        from convs.linears import OLF as OLF

        self.olf_layer = OLF(in_features=in_dim, out_features=out_dim, W0_torch=W0.T, rank=self.hidden_dim_t)

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
        input_features = self.visual_forward_(x)
        norm_input_features = input_features / input_features.norm(dim=-1, keepdim=True)

        cls_results = []
        for cls in self.classifier_list:
            cls_results.append(cls(norm_input_features))

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


    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("updating stat")
        with torch.no_grad():
            vecs = []
            vecs_norm = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                image_features = self.visual_forward_(inputs)
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                vecs.append(image_features)
                vecs_norm.append(image_features_norm)
                labels.append(targets)

        for i in range(known_classes, total_classes):
            self.label2task[i] = self.task_id

        vecs = torch.cat(vecs)
        self.olf_layer.update_old_features(vecs)
        vecs_norm = torch.cat(vecs_norm)
        labels = torch.cat(labels)

        mu = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)], dim=0)
        center_vecs = torch.cat([vecs[labels == i] - mu[i - known_classes] for i in range(known_classes, total_classes)], dim=0)
        cov = torch.cov(center_vecs.t()) + 1e-4 * torch.eye(center_vecs.shape[-1]).to(device)

        mu_norm = torch.cat([vecs_norm[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)], dim=0)
        center_vecs_norm = torch.cat([vecs_norm[labels == i] - mu_norm[i - known_classes]
                                     for i in range(known_classes, total_classes)], dim=0)
        cov_inv = center_vecs_norm.shape[1] * torch.linalg.pinv(
            (center_vecs_norm.shape[0] - 1) * center_vecs_norm.T.cov() + center_vecs_norm.T.cov().trace() * torch.eye(center_vecs_norm.shape[1]).cuda())
        current_ps = torch.ones(mu_norm.shape[0]).cuda() * 1. / mu_norm.shape[0]
        self.current_W = torch.einsum('nd, dc -> cn', mu_norm, cov_inv)
        self.current_b = current_ps.log() - torch.einsum('nd, dc, nc -> n', mu_norm, cov_inv, mu_norm) / 2

        if self.mu is None:
            self.mu = mu
            self.mu_norm = mu_norm
            self.cov_inv = cov_inv
            self.cov_list = [cov]
            self.update_cov = cov
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

        ps = torch.ones(self.mu_norm.shape[0]).cuda() * 1. / self.mu_norm.shape[0]
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
            return self.olf_layer(self.mu)

    def get_cls_center_lora(self):
        with torch.no_grad():
            training_state = self.olf_layer.training
            self.olf_layer.eval()
            new_center = self.olf_layer(self.mu)
            self.olf_layer.train(training_state)
        return new_center

    def get_param_group(self):
        param_groups = []
        param_groups.append({'params': self.olf_layer.get_trainable_parameters()})
        param_groups.append({'params': self.olf_layer.get_stage2_parameters(), 'lr': 0.001, 'weight_decay': 0.001})

        if len(self.classifier_list) > 0:
            param_groups.append({'params': self.classifier_list[-1].parameters(),
                                'lr': 0.001, 'weight_decay': 0.001})

        return param_groups

    def train_state(self):
        self.olf_layer.train()

    def eval_state(self):
        self.olf_layer.eval()
