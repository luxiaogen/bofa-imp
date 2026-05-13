import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


class OLF(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            W0_torch,
            rank=64,
            subspace_policy="data_oss",
            basis_seed=1993,
            basis_alloc="disjoint_block",
            ####################add-4.28-start######################
            shared_rank=-1,
            shared_lr_scale=0.1,
            ####################add-4.28-end########################
            first_task_rank=-1, # add-5.3
            ####################add-5.5-start######################
            shared_importance_mode="none",
            importance_beta=0.9,
            importance_alpha=1.0,
            ####################add-5.5-end######################
            ####################add-5.7-start######################
            shared_svd_reg_lambda=0.0,
            shared_svd_reg_topk=20,
            ####################add-5.7-end#########################
            ####################add-5.7-start######################
            shared_svd_grad_mode="none",
            ####################add-5.7-end#########################
            ####################add-5.8-start######################
            shared_param_reg_mode="none",
            shared_param_reg_lambda=0.0,
            shared_param_importance_beta=0.9,
            ####################add-5.8-end#########################
    ):
        super().__init__()
        self.in_features = in_features # 768
        self.out_features = out_features # 512
        self.rank = rank # 256
        #self.current_rank = 0  # 当前任务的秩
        #W0_torch = W0_torch.cuda()  # 确保W0在GPU上


        ####################add-4.27-start######################
        self.current_rank = 0
        self.subspace_policy = subspace_policy # 子空间的策略
        self.basis_seed = basis_seed # 1993
        self.basis_alloc = basis_alloc # disjoint_block 每个任务使用互不重叠的基底子空间 任务独立，避免遗忘 | 'front_loaded_block'

        ####################add-4.28-start######################
        self.shared_lr_scale = shared_lr_scale # 0.1
        ####################add-4.28-end########################
        ####################add-5.5-start######################
        self.shared_importance_mode = shared_importance_mode # 'column_grad_scale'
        self.importance_beta = importance_beta # 0.9
        self.importance_alpha = importance_alpha # 1.0
        self.importance_eps = 1e-8
        ####################add-5.5-end######################
        self.first_task_rank = first_task_rank # add-5.3
        ####################add-5.7-start######################
        self.shared_svd_reg_lambda = shared_svd_reg_lambda # 0.001
        self.shared_svd_reg_topk = shared_svd_reg_topk # 20
        self.shared_svd_grad_mode = shared_svd_grad_mode # 'ogd_project'
        ####################add-5.7-end#########################
        ####################add-5.8-start######################
        self.shared_param_reg_mode = shared_param_reg_mode
        self.shared_param_reg_lambda = shared_param_reg_lambda
        self.shared_param_importance_beta = shared_param_importance_beta
        ####################add-5.8-end#########################
        self.total_tasks = None

        self.active_slice = None
        #target_device = W0_torch.device if W0_torch.is_cuda else torch.device(
            #"cuda" if torch.cuda.is_available() else "cpu")
        W0_torch = W0_torch.cuda()
        ####################add-4.27-end########################
        self.active_rank = rank # add-5.3

        self.register_buffer('W0', W0_torch.clone().detach())

        ####################add-4.28-start######################
        if self.subspace_policy == "fixed_svd_shared_core":
            self.shared_rank = rank // 2 if shared_rank is None or shared_rank < 0 else shared_rank # 32 | 468 共享空间
            self.private_rank = rank - self.shared_rank # 32 | 30 私有空间
            # if self.shared_rank <= 0 or self.private_rank <= 0:
            #     raise ValueError(
            #         f"fixed_svd_shared_core requires 0 < shared_rank < rank, got shared_rank={self.shared_rank}, rank={rank}."
            #     )
        else:
            self.shared_rank = 0
            self.private_rank = rank
        ####################add-4.28-end########################
        # self.rank_capacity 768
        self.rank_capacity = self.in_features if self._uses_front_loaded_block() else self.rank # add-5.3
        self.W_task = nn.Parameter(W0_torch.clone()) # [512,768] 任务特定权重
        # self.B = nn.Parameter(torch.zeros(self.out_features, self.rank)) # [256,512]  [512,rank=64]
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank_capacity)) # [512,498] mod-5.3

        # self.A = torch.zeros(self.in_features, self.rank) # [768,256]
        ####################add-4.27-start######################
        # [768,498]
        self.register_buffer('A', torch.zeros(self.in_features, self.rank_capacity, device=self.W0.device)) # mod-5.3
        self.register_buffer('fixed_basis', torch.empty(self.in_features, 0, device=self.W0.device))
        ####################add-4.27-end########################
        ####################add-4.28-start######################
        self.B_shared = nn.Parameter(torch.zeros(self.out_features, self.shared_rank, device=self.W0.device)) # [512, 468]：所有任务共享，持续更新
        self.B_private = nn.Parameter(torch.zeros(self.out_features, self.private_rank, device=self.W0.device)) # [512, 30]：每个任务独有，任务切换时清零
        self.register_buffer('shared_basis_block',torch.empty(self.in_features, self.shared_rank, device=self.W0.device)) # 正交基 [768, 468]
        self.register_buffer('private_basis_block',torch.empty(self.in_features, self.private_rank, device=self.W0.device)) # 正交基 [768, 30]
        ####################add-4.28-end########################
        ####################add-5.5-start######################
        self.register_buffer('shared_importance', torch.zeros(self.shared_rank, device=self.W0.device)) # [468]
        self.register_buffer('shared_grad_scale', torch.ones(self.shared_rank, device=self.W0.device)) # [468]
        ####################add-5.5-end######################
        ####################add-5.7 svd loss + grad svd-start######################
        self.shared_svd_topk = min(int(self.shared_svd_reg_topk), self.out_features, self.shared_rank) # min(20,512,468)
        self.register_buffer('shared_svd_anchor_B', # [512, 468] 锚点（参考点），通常是上一个任务结束时的 B_shared
                             torch.zeros(self.out_features, self.shared_rank, device=self.W0.device))
        self.register_buffer('shared_svd_U', # [512, 20]
                             torch.empty(self.out_features, self.shared_svd_topk, device=self.W0.device))
        self.register_buffer('shared_svd_V', torch.empty(self.shared_rank, self.shared_svd_topk, device=self.W0.device)) # [468, 20]
        self.register_buffer('shared_svd_weight', torch.empty(self.shared_svd_topk, device=self.W0.device)) # [20]
        self.register_buffer('shared_svd_ready', torch.tensor(False, device=self.W0.device))
        ####################add-5.7-end#########################
        ####################add-5.8-start######################
        self.register_buffer('shared_param_anchor_B',
                             torch.zeros(self.out_features, self.shared_rank, device=self.W0.device))
        self.register_buffer('shared_param_importance',
                             torch.zeros(self.out_features, self.shared_rank, device=self.W0.device))
        self.register_buffer('shared_param_importance_accum',
                             torch.zeros(self.out_features, self.shared_rank, device=self.W0.device))
        self.register_buffer('shared_param_importance_steps', torch.tensor(0.0, device=self.W0.device))
        self.register_buffer('shared_param_ready', torch.tensor(False, device=self.W0.device))
        ####################add-5.8-end#########################

        self.W_task.requires_grad = False  # 初始时冻结
        self.W_fusion = W0_torch.clone().detach() # [512,768] mean(所有任务权重)
        self.W_fusion2 = W0_torch.clone().detach() # mean(所有任务权重 + 原始CLIP权重)      # 包含CLIP
        self.cov_matrices = []

        self.W_list = []
        self.task_id = 0
        self.start_eval = True
        ####################add-4.27######################
        if self._uses_fixed_basis(): #  检查是否使用固定基底
            self._init_fixed_basis()

    def forward(self, x, stage2=False):
        # x[bs,768]
        if self.training:
            if self._uses_fixed_basis(): # 使用固定正交基 add-4.27
                return F.linear(x, self._compose_task_weight()) # [20,768]@[768,512]=[20,512]
            # 在训练时，使用当前的可训练参数 W_task
            if stage2:
                return F.linear(x, self.W0+self.B @ self.A.T) # [512,768]+[512,256]@[256,768]=[512,768]
            # else:
            return F.linear(x, self.W_task)
        else:
            return F.linear(x, self.W_fusion)

    def eval_forward(self, x):
        # 1. 使用 W_fusion（纯任务平均）   2. 使用 W_fusion2（任务 + 原始 CLIP）
        # W_fusion  = mean(任务权重) = 只有训练任务的知识
        # W_fusion2 = mean(任务权重 + W0) = 训练任务 + CLIP 预训练的知识 ✅
        return (F.linear(x, self.W_fusion), F.linear(x, self.W_fusion2))

    def update_old_features(self, features):
        ####################add-4.27-start######################
        if not self.uses_data_oss():
            return
        ####################add-4.27-end########################
        mean = torch.mean(features, dim=0, keepdim=True) # num_samples mean [1,768]
        centered_features = features - mean # [5000,768] remove central

        k = centered_features.shape[0]
        cov = (1 / (k - 1)) * (features.T @ features) # [768,768] 无偏估计(用样本计算了均值，已经"用掉"了 1 个自由度，所以要除以 n-1)
        reg = 1e-4 * torch.eye(self.in_features, device=self.W0.device) # 加正则化 防止矩阵奇异(矩阵不可逆，行列式为 0)
        # 可以安全地求逆了 cov_inv = torch.inverse(cov_reg)
        self.cov_matrices.append((cov + reg).detach())
    # 任务 1+ (后续任务)
    def prepare_for_new_task(self):

        self.train()
        # self. start_eval = False
        # print(f"OLF: Preparing for Task {self.task_id}. Unfreezing W_task for fine-tuning.")
        # 解冻 W_task 用于 Stage 1
        # self.W_task.requires_grad = True
        ####################add-4.27-start######################
        self.start_eval = False
        if self._uses_fixed_basis(): # 策略: 使用固定基
            # basis_start, basis_end = self._assign_task_subspace() # # 1. 分配子空间  # 例如: Task 0 → [0:64), Task 1 → [64:128), ...
            # self.B.data.zero_() # 2. 初始化 B 为 0
            # self.B.requires_grad = True # 3. 训练 B，冻结 W_task

            ####################add-4.28-start######################
            basis_info = self._assign_task_subspace() # (0,32,32,64) | (0,468,468,498)
            if self._uses_shared_core():
                _, _, private_start, private_end = basis_info # 32,64
                self.B_private.data.zero_()
                self.B_shared.requires_grad = True
                self.B_private.requires_grad = True
            else:
                basis_start, basis_end = basis_info
                self.B.data.zero_()
                self.B.requires_grad = True
            ####################add-4.28-end########################

            self.W_task.requires_grad = False

        else: # 策略2: 使用 OSS 子空间（原始 BOFA）
            print(f"OLF: Preparing for Task {self.task_id}. Unfreezing W_task for fine-tuning.")
            self.W_task.requires_grad = True
        ####################add-4.27-end########################
    #    Step 3: 计算 OSS (进入 Stage 2)
    def prepare_for_stage2(self):
        ####################add-4.27-start######################
        if self._uses_fixed_basis():
            print(f"OLF[{self.subspace_policy}]: Stage 2 skipped; fixed basis is active from task start.")
            return
        ####################add-4.27-end########################
        self.train()
        self.start_eval = False
        print(f"OLF: Preparing for Stage 2 of Task {self.task_id}. Computing OSS and initializing B.")

        if self.task_id > 0:
            # 1. 聚合旧任务协方差矩阵 Σ=(1/(n-1))* X^T@X  | Task 2: [cov_task0(反映 Task 0 数据的分布), cov_task1(反映 Task 1 数据的分布)]
            covs_to_aggregate = self.cov_matrices[:-1] # [768,768]
            if len(covs_to_aggregate) > 0:  # avg_cov = (cov_0 + cov_1) / 2(反映 Task 0 和 Task 1 的共同数据分布)
                avg_cov = torch.mean(torch.stack(covs_to_aggregate, dim=0), dim=0) # 对任务求平均 -> d
                # 2. 特征值分解 对协方差矩阵做特征值分解 | 对厄米矩阵（Hermitian）或对称矩阵进行特征值分解
                eigenvalues, eigenvectors = torch.linalg.eigh(avg_cov) # [768]  [768,768]
                # eigenvalues: [768] - 每个特征方向的"重要性" | 特征值保证是实数 | 从小到大排序 ✅
                # eigenvectors: [768, 768] - 每列是一个特征方向 特征向量: 在特征空间中的方向 ✅

                # 3. 选择主子空间
                self.A.data = eigenvectors[:, :self.rank].clone() # [768,low-rand=256]
        else:
            pass  # A 保持为零，阶段二对task 0无效
        """
            - `A` 张成旧任务的主要变化方向
            - `B` 在这个子空间中调整新任务
            - 由于 `A` 正交于旧任务的其他方向，不会干扰旧知识
        """
        delta_W_tilde = self.W_task.data - self.W0 # [512,768] ΔW=W_new - W_old
        with torch.no_grad():
            self.B.data = delta_W_tilde @ self.A # 4. 初始化 B
        # 5. 冻结 W_task，只训练 B
        self.W_task.requires_grad = False
        self.B.requires_grad = True # train B
    # Step 5: 结束任务
    def end_task(self):

        print(f"OLF: Ending Task {self.task_id}.")
        self.W_task.requires_grad = False
        self.B.requires_grad = False # add-4.27
        ####################add-4.28-start######################
        self.B_shared.requires_grad = False
        self.B_private.requires_grad = False
        ####################add-4.28-end########################


        # 1. 保存当前任务权重
        if self._uses_fixed_basis(): # add-4.27
            self.W_list.append(self._compose_task_weight().detach().clone()) # add-4.27
        elif self.task_id == 0:
            self.W_list.append(self.W_task.data.clone().detach())
        else:
            diff_W = self.B @ self.A.T
            self.W_list.append(self.W0 + diff_W.detach().clone())
        # 2. 融合所有任务权重
        self.W_fusion = (sum(self.W_list)) / len(self.W_list)
        # 3. 另一种融合方式（包含 W0）
        self.W_fusion2 = (sum(self.W_list)+self.W0) / (len(self.W_list)+1)

        ####################add-5.5-start######################
        self.update_shared_importance()
        ####################add-5.5-end######################
        ####################add-5.7-start######################
        self.update_shared_svd_anchor()
        ####################add-5.7-end#########################
        ####################add-5.8-start######################
        self.update_shared_param_importance()
        ####################add-5.8-end#########################

        self.task_id += 1
        # self.current_rank += self.rank
        self.current_rank += self.active_rank # mod-5.3
        self.start_eval = True
        self.eval()

    def get_trainable_parameters(self):
        ####################add-4.28-start######################
        if self._uses_shared_core():
            return []
        ####################add-4.28-start######################
        ####################add-4.27-start######################
        if self._uses_fixed_basis():
            return [self.B]
        ####################add-4.27-end########################
        return [self.W_task]

    def get_stage2_parameters(self):
        ####################add-4.27-start######################
        if self._uses_fixed_basis():
            return []
        ####################add-4.27-end########################
        return [self.B]

    def get_projected_weight(self, W_t):
        covs_to_aggregate = self.cov_matrices[:-1]
        avg_cov = torch.mean(torch.stack(covs_to_aggregate, dim=0), dim=0)

        _, V = torch.linalg.eigh(avg_cov)
        Unull = V[:, :self.rank]

        P = Unull @ Unull.T
        projected_W = W_t @ P
        return projected_W
    ####################add-4.27-start######################
    def uses_data_oss(self):
        return self.subspace_policy == "data_oss"

    def uses_two_stage(self):
        return self.uses_data_oss()

    ####################add-4.28-start######################
    def set_total_tasks(self, total_tasks):
        self.total_tasks = total_tasks

        ##########add.5.3-start######################
        # if self._uses_front_loaded_block(): # True
        #     first_rank = self._get_task_rank(0) # 768-30*9=498
        #     required = first_rank + (total_tasks - 1) * self.rank # 768
            # if first_rank <= 0 or required > self.in_features:
            #     raise ValueError(
            #         f"Front-loaded capacity exceeded: total_tasks={total_tasks}, tail_rank={self.rank}, "
            #         f"computed first_task_rank={first_rank}, required={required}, available={self.in_features}."
            #     )
        ##########add.5.3-end######################

    def uses_shared_core(self):
        return self._uses_shared_core()
    ####################add-4.28-end######################

    def _uses_fixed_basis(self):
        #return self.subspace_policy in {"fixed_svd_basis", "fixed_fullrank_spectrum"}
        return self.subspace_policy in {"fixed_svd_basis", "fixed_svd_shared_core"}

    ####################add-4.28-start######################
    def _uses_shared_core(self):
        return self.subspace_policy == "fixed_svd_shared_core"

    def get_shared_parameters(self):
        if not self._uses_shared_core():
            return []
        return [self.B_shared]

    def get_private_parameters(self):
        if not self._uses_shared_core():
            return []
        return [self.B_private]
    ####################add-4.28-end######################

    def _compose_task_weight(self):
        ####################add-4.28-start######################
        if self._uses_shared_core(): # A-> shared+privated  A：方向基底 / 子空间(“允许沿哪些方向更新？”,A 决定“空间”)  B：这些方向的系数(“每个方向更新多少？”,B 决定“坐标”)
            shared_weight = self.B_shared @ self.shared_basis_block.T # 共享方向基 共享系数矩阵[512, 32] @ 共享子空间基[32, 768] = [512, 768]
            private_weight = self.B_private @ self.private_basis_block.T # 私有方向基  私有系数矩阵[512, 32] @ 私有子空间基[32, 768] = [512, 768]
            return self.W0 + shared_weight + private_weight # 最终权重 = 原始权重 + 共享贡献 + 私有贡献
        ####################add-4.28-end######################f
        # return self.W0 + self.B @ self.A.T
        return self.W0 + self.B[:, :self.active_rank] @ self.A[:, :self.active_rank].T # mod-5.3
    # 固定基方法 直接训练 B:W = W0 + B @ A^T   其中 A 是固定的随机正交基
    def _assign_task_subspace(self):
        #if self.total_tasks is None:
            #raise RuntimeError("set_total_tasks must be called before fixed-basis training starts.")
        ####################add-4.28-start######################
        if self._uses_shared_core():
            private_start = self.shared_rank + self.task_id * self.private_rank # 32 任务独享的方向
            private_end = private_start + self.private_rank # 64

            self.shared_basis_block.copy_(self.fixed_basis[:, :self.shared_rank]) # [768,32]
            self.private_basis_block.copy_(self.fixed_basis[:, private_start:private_end]) # [768,32]
            self.A.copy_(torch.cat((self.shared_basis_block, self.private_basis_block), dim=1)) # [768,32(共享的)+32(私有的)]
            self.active_rank = self.shared_rank + self.private_rank #  截止到当前任务需要激活的秩  468+30
            self.active_slice = (0, self.shared_rank, private_start, private_end)
            return self.active_slice
        ####################add-4.28-end########################
        # start = self.task_id * self.rank
        # end = start + self.rank
        ####################add-5.3-start########################
        start, end = self._get_task_slice(self.task_id) # task0: 0 498
        active_rank = end - start # task0:498
        # if end > self.in_features:
        #     raise ValueError(
        #         f"Task {self.task_id} exceeds fixed-basis capacity: need directions [{start}:{end}), "
        #         f"but only {self.in_features} are available."
        #     )

        basis_block = self.fixed_basis[:, start:end] # task0:768,498

        # self.A.copy_(basis_block)
        self.A.zero_()
        self.A[:, :active_rank].copy_(basis_block)
        self.active_rank = active_rank
        self.active_slice = (start, end)
        return start, end
        ####################add-5.3-end########################

        # basis_block = self.fixed_basis[:, start:end] # [768,64]   特征向量
        # if self.subspace_policy == "fixed_fullrank_spectrum": #  如果使用 fullrank_spectrum，用特征值缩放
        #     spectrum_block = self.fixed_spectrum[start:end].unsqueeze(0) # [1,low_rank=64]
        #     basis_block = basis_block * spectrum_block # tezhengxiangliang * tezhegnzhi 重要的方向（大特征值）贡献更多 更接近真实的协方差矩阵结构
        # self.A.copy_(basis_block) # 3. 保存到 A
        # self.active_slice = (start, end) # 记录当前任务使用的子空间范围
        # return start, end
    # 初始化一个固定的正交基（用于替代 OSS 子空间）
    def _init_fixed_basis(self):
        #if self.basis_alloc != "disjoint_block":
            #raise ValueError(f"Unsupported basis_alloc: {self.basis_alloc}")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.basis_seed) # 固定随机种子
        random_matrix = torch.randn(
            self.in_features, # 768
            self.in_features, # 768
            generator=generator,
            dtype=self.W0.dtype,  # 元素服从 N(0,1)
        ) # [768, 768]

        #if self.subspace_policy == "fixed_svd_basis": # 对随机矩阵做 SVD
        if self.subspace_policy in {"fixed_svd_basis", "fixed_svd_shared_core"}:
            # U, _, _ = torch.linalg.svd(random_matrix, full_matrices=True)
            U, S, Vh = torch.linalg.svd(random_matrix, full_matrices=True) # 需要完整的正交基时，比如某些理论分析、要保证 U 和 V 是标准正交方阵，或者后续运算要求 U 必须是方阵
            self.fixed_basis = U.to(self.W0.device) # U: [768, 768] [768] [768,768] - 正交矩阵 # 正交基
            self.fixed_spectrum = torch.ones(self.in_features, device=self.W0.device, dtype=self.W0.dtype) # 所有特征值都是 1

        # ortho_error = torch.max(
        #     torch.abs(
        #         self.fixed_basis.T @ self.fixed_basis - torch.eye(self.in_features, device=self.fixed_basis.device))
        # ).item() # 正交矩阵应该满足: U^T @ U = I ortho_error 应该接近 0
        # print(ortho_error)


    ####################add-4.27-end########################
    ##########add.5.3-start######################
    def _uses_front_loaded_block(self):
        return self._uses_fixed_basis() and not self._uses_shared_core() and self.basis_alloc == "front_loaded_block"

    def _get_task_rank(self, task_id):
        if not self._uses_front_loaded_block():
            return self.rank
        # if self.total_tasks is None:
        #     raise RuntimeError("set_total_tasks must be called before computing front-loaded ranks.")
        if task_id == 0:
            if self.first_task_rank is not None and self.first_task_rank > 0:
                return self.first_task_rank
            return self.in_features - (self.total_tasks - 1) * self.rank
        return self.rank

    def _get_task_slice(self, task_id):
        if not self._uses_front_loaded_block():
            start = task_id * self.rank
            end = start + self.rank
            return start, end

        first_rank = self._get_task_rank(0)
        if task_id == 0:
            return 0, first_rank
        start = first_rank + (task_id - 1) * self.rank
        end = start + self.rank
        return start, end
    ####################mod-5.3-end###########################
    ##########add.5.3-end######################
    ##########add.5.4-vis-start######################
    def get_b_snapshot(self):
        snapshot = {
            "task_id": self.task_id,
            "subspace_policy": self.subspace_policy,
            "basis_alloc": self.basis_alloc,
            "active_rank": int(self.active_rank),
            "active_slice": self.active_slice,
        }
        if self._uses_shared_core():
            snapshot.update(
                {
                    "B_shared": self.B_shared.detach().cpu().clone(),
                    "B_private": self.B_private.detach().cpu().clone(),
                    "shared_rank": int(self.shared_rank),
                    "private_rank": int(self.private_rank),
                    "private_slice": self.active_slice,
                    ####################add-5.5-start######################
                    "shared_importance_mode": self.shared_importance_mode,
                    "shared_importance": self.shared_importance.detach().cpu().clone(),
                    "shared_grad_scale": self.shared_grad_scale.detach().cpu().clone(),
                    ####################add-5.5-end######################
                    ####################add-5.7-start######################
                    "shared_svd_reg_lambda": float(self.shared_svd_reg_lambda),
                    "shared_svd_reg_topk": int(self.shared_svd_topk),
                    "shared_svd_grad_mode": self.shared_svd_grad_mode,
                    "shared_svd_ready": bool(self.shared_svd_ready.item()),
                    "shared_svd_weight": self.shared_svd_weight.detach().cpu().clone(),
                    ####################add-5.7-end#########################
                    ####################add-5.8-start######################
                    "shared_param_reg_mode": self.shared_param_reg_mode,
                    "shared_param_reg_lambda": float(self.shared_param_reg_lambda),
                    "shared_param_ready": bool(self.shared_param_ready.item()),
                    "shared_param_importance": self.shared_param_importance.detach().cpu().clone(),
                    ####################add-5.8-end#########################
                }
            )
        else:
            snapshot.update(
                {
                    "B": self.B[:, :self.active_rank].detach().cpu().clone(),
                    "slice": self.active_slice,
                }
            )
        return snapshot
    ##########add.5.4-vis-end######################
    ####################add-5.5-start######################
    def _uses_shared_importance(self):
        return self._uses_shared_core() and self.shared_importance_mode == "column_grad_scale"

    def apply_shared_importance_to_grads(self):
        if not self._uses_shared_importance():
            return
        if self.task_id == 0:
            return
        if self.B_shared.grad is None or self.shared_grad_scale.numel() == 0:
            return # [512,468]@[1,468] | shared_grad_scale:一个缩放因子张量，用于调整不同维度的梯度大小
        self.B_shared.grad.mul_(self.shared_grad_scale.unsqueeze(0))

    def update_shared_importance(self):
        if not self._uses_shared_importance():
            return
        if self.shared_rank <= 0:
            return
        with torch.no_grad():
            # 步骤1: 计算每列(竖着算)的L2范数（B_shared[:, j] 很大，说明模型在第 j 个共享方向上学到了较强的修正量，也就是这个方向被更多使用。反过来，如果这一列接近 0，说明这个共享方向几乎没贡献）
            col_norm = torch.norm(self.B_shared.detach(), dim=0) # [512,468]->[468] | col_norm[j] = sqrt(Σ_i B_shared[i, j]^2)
            # 步骤2: 指数移动平均更新重要性
            if self.task_id == 0 and torch.count_nonzero(self.shared_importance).item() == 0:
                updated_importance = col_norm # 初始化
            else: # importance_beta:EMA系数 importance_alpha  平滑地累积历史重要性，避免剧烈波动
                updated_importance = self.importance_beta * self.shared_importance + (
                            1 - self.importance_beta) * col_norm # 468  0.9*shared_importance+0.1*col_norm | 90%权重给历史（保持稳定）+ 10%权重给当前（适应新任务）
            self.shared_importance.copy_(updated_importance)
            # 步骤3: 归一化重要性
            mean_importance = updated_importance.mean() # danshu zhi 计算所有维度重要性的均值
            norm_importance = updated_importance / (mean_importance + self.importance_eps) # [468]
            # 步骤4: 计算梯度缩放因子（重要性越高，梯度缩放越小） importance_alpha:重要性缩放系数
            # grad_scale = 1.0 / (1.0 + self.importance_alpha * norm_importance) # 保护这个知识不被后续任务破坏
            # 步骤5: 二次归一化   反比例缩放
            grad_scale = 1.0 / (1.0 + self.importance_alpha * norm_importance)
            # grad_scale = grad_scale / (grad_scale.mean() + self.importance_eps) # [468]
            # grad_scale = grad_scale / (grad_scale.mean() + self.importance_eps)
            # self.shared_grad_scale.copy_(grad_scale.clamp_(0.0, 2.0))
            self.shared_grad_scale.copy_(grad_scale.clamp_(0.0, 1.0))
    ####################add-5.5-end######################
    ####################add-5.7-start######################
    def _uses_shared_svd_regularization(self):
        return self._uses_shared_core() and self.shared_svd_reg_lambda > 0 and self.shared_svd_topk > 0

    def _uses_shared_svd_ogd(self):
        return self._uses_shared_core() and self.shared_svd_grad_mode == "ogd_project" and self.shared_svd_topk > 0

    def _uses_shared_svd_anchor(self):
        return self._uses_shared_svd_regularization() or self._uses_shared_svd_ogd()

    def update_shared_svd_anchor(self):
        if not self._uses_shared_svd_anchor():
            return
        with torch.no_grad(): # 任务结束时，保存当前 B_shared 的 top-k SVD 子空间
            anchor = self.B_shared.detach() # 旧任务的 B_shared
            U, S, Vh = torch.linalg.svd(anchor, full_matrices=False) # old_shared_B
            topk = self.shared_svd_topk # 预设的 20-468
            weights = S[:topk] / (S[:topk].mean() + self.importance_eps) # 获取相应的权重
            self.shared_svd_anchor_B.copy_(anchor)
            self.shared_svd_U.copy_(U[:, :topk])
            self.shared_svd_V.copy_(Vh[:topk, :].T)
            self.shared_svd_weight.copy_(weights)
            self.shared_svd_ready.fill_(True)

    # 后续任务训练时，计算 B_shared 相对旧 anchor 的变化，并投影到旧 top-k 子空间里惩罚
    def shared_svd_regularization(self):
        if not self._uses_shared_svd_regularization():
            return self.B_shared.new_tensor(0.0)  # 如果没有使用 svd 正则化
        if self.task_id == 0 or not bool(self.shared_svd_ready.item()):
            return self.B_shared.new_tensor(0.0)
        delta = self.B_shared - self.shared_svd_anchor_B # [512,468] 参数相对于某个"安全状态"偏移了多少
        # [20,20] U.T 和 V 是旋转矩阵，将坐标系旋转到主成分方向 U.T @ delta:将 delta 投影到输出空间的主方向 @ V：再投影到输入空间的主方向    将变化量投影到SVD子空间
        projected_delta = self.shared_svd_U.T @ delta @ self.shared_svd_V
        # 重要方向（大奇异值）的变化会被赋予更大的惩罚权重,不重要方向（小奇异值）的变化惩罚较小
        # [20,20] 构造加权矩阵, shared_svd_weight[20] svd value M[i][j] = weight[i] * weight[j]
        # torch.outer(a, b) 矩阵中位置 (i, j) 的元素 = a[i] * b[j]
        weight_matrix = torch.sqrt(torch.outer(self.shared_svd_weight, self.shared_svd_weight)) #
        # return self.shared_svd_reg_lambda * torch.mean((projected_delta * weight_matrix) ** 2) # 0.001
        # 平均 topkxtop 后的效果不行 换成 /top 的
        # 原来: sum / (topk * topk)
        # 现在: sum / topk
        topk = max(1, projected_delta.shape[0]) # topk
        return self.shared_svd_reg_lambda * torch.sum(
            (projected_delta * weight_matrix) ** 2 # F 范数，本质上是为了度量这个矩阵整体变化有多大
        ) / topk

    def apply_shared_svd_ogd_to_grads(self):
        if not self._uses_shared_svd_ogd():
            return
        if self.task_id == 0 or not bool(self.shared_svd_ready.item()): # Task 0 不需要保护（没有旧知识）
            return
        if self.B_shared.grad is None:
            return
        with torch.no_grad():
            """
                # 原始梯度
                grad = ∇_parallel + ∇_perp
                
                # 计算平行分量（投影）
                protected_grad = ∇_parallel = P(grad)
                
                # 移除平行分量，只保留正交分量
                grad = grad - protected_grad = ∇_perp
            """
            grad = self.B_shared.grad # [512,468] 反向传播计算出的原始梯度
            # 计算需要保护的梯度分量
            ## Step 1: 投影到SVD子空间 self.shared_svd_U.T @ grad @ self.shared_svd_V
            ## [topk, 512] @ [512, 468] @ [468, topk] = [topk, topk] UV,旧任务的重要方向
            ### self.shared_svd_U.copy_(U[:, :topk])
            ### self.shared_svd_V.copy_(Vh[:topk, :].T)
            projected = self.shared_svd_U.T @ grad @ self.shared_svd_V # [20,20] 在重要方向上的梯度
            ## Step 2: 投影回原始空间
            ## [512, topk] @ [topk, topk] @ [topk, 468] = [512, 468]
            protected_grad = self.shared_svd_U @ projected @ self.shared_svd_V.T
            grad.sub_(protected_grad) # 从梯度中减去保护分量,移除了会影响旧任务的梯度分量,只保留与旧任务正交的梯度分量,参数更新只在"安全方向"上进行
    ####################add-5.7-end#########################
    ####################add-5.8-start######################
    # 检查是否启用EWC正则化
    def _uses_shared_param_regularization(self):
        return (
                self._uses_shared_core() # 使用共享B
                and self.shared_param_reg_mode == "ewc_grad"
                and self.shared_param_reg_lambda > 0 # 正则化强度
                and self.shared_rank > 0 # 共享维度数
        )

    # 计算EWC正则化损失
    def shared_param_regularization(self):
        if not self._uses_shared_param_regularization():
            return self.B_shared.new_tensor(0.0) # 不启用EWC，返回0损失
        if self.task_id == 0 or not bool(self.shared_param_ready.item()):
            return self.B_shared.new_tensor(0.0) # Task 0：没有旧任务需要保护
        delta = self.B_shared - self.shared_param_anchor_B # shared_param_anchor_B：上一个任务结束时的参数（锚点）
        return self.shared_param_reg_lambda * torch.sum(self.shared_param_importance * delta.pow(2))

    # 累积参数重要性（在训练过程中调用）
    # EWC 先在训练时累计 B_shared.grad^2，作为参数重要性估计
    def accumulate_shared_param_importance(self):
        if not self._uses_shared_param_regularization():
            return
        if self.B_shared.grad is None:
            return
        with torch.no_grad(): # accum←accum+∇^2 -- 梯度平方越大，说明参数对损失函数越敏感，越重要
            self.shared_param_importance_accum.add_(self.B_shared.grad.detach().pow(2))
            self.shared_param_importance_steps.add_(1.0) # 记录累积了多少次梯度
    # 任务结束时更新重要性（在 end_task() 中调用）
    def update_shared_param_importance(self):
        if not self._uses_shared_param_regularization(): #
            return
        if self.shared_param_importance_steps.item() <= 0:
            self.shared_param_anchor_B.copy_(self.B_shared.detach()) # 直接保存当前参数作为锚点
            self.shared_param_ready.fill_(True)
            return
        with torch.no_grad():
            task_importance = self.shared_param_importance_accum / ( # 计算当前任务的平均梯度平方-->参数在当前任务中的平均敏感度
                        self.shared_param_importance_steps + self.importance_eps)
            task_importance = task_importance / (task_importance.mean() + self.importance_eps)
            if not bool(self.shared_param_ready.item()) or torch.count_nonzero(
                    self.shared_param_importance).item() == 0:
                updated_importance = task_importance
            else: # 指数移动平均（Task 1+） 平滑地累积多个任务的重要性
                updated_importance = (
                        self.shared_param_importance_beta * self.shared_param_importance
                        + (1 - self.shared_param_importance_beta) * task_importance
                ) # EMA融合
            updated_importance = updated_importance / (updated_importance.mean() + self.importance_eps)
            self.shared_param_importance.copy_(updated_importance) # 更新
            self.shared_param_anchor_B.copy_(self.B_shared.detach())
            self.shared_param_importance_accum.zero_()
            self.shared_param_importance_steps.zero_()
            self.shared_param_ready.fill_(True)
    ####################add-5.8-end#########################