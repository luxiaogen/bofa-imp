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
            basis_eps=1e-4,
            basis_zero_fix="near_zero_only",
            ####################add-4.28-start######################
            shared_rank=-1,
            shared_lr_scale=0.1,
            ####################add-4.28-end########################
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
        self.basis_alloc = basis_alloc # disjoint_block 每个任务使用互不重叠的基底子空间 任务独立，避免遗忘
        self.basis_eps = basis_eps # 0.001 修复接近零的特征值，避免数值不稳定
        self.basis_zero_fix = basis_zero_fix # nero_zero_only 特征值修复策略
        ####################add-4.28-start######################
        self.shared_lr_scale = shared_lr_scale # 0.1
        ####################add-4.28-end########################
        self.total_tasks = None
        self.fixed_basis_fixes = 0 # 记录修复的特征值数量
        self.active_slice = None
        #target_device = W0_torch.device if W0_torch.is_cuda else torch.device(
            #"cuda" if torch.cuda.is_available() else "cpu")
        W0_torch = W0_torch.cuda()
        ####################add-4.27-end########################

        self.register_buffer('W0', W0_torch.clone().detach())

        ####################add-4.28-start######################
        if self.subspace_policy == "fixed_svd_shared_core":
            self.shared_rank = rank // 2 if shared_rank is None or shared_rank < 0 else shared_rank # 32
            self.private_rank = rank - self.shared_rank # 32
            # if self.shared_rank <= 0 or self.private_rank <= 0:
            #     raise ValueError(
            #         f"fixed_svd_shared_core requires 0 < shared_rank < rank, got shared_rank={self.shared_rank}, rank={rank}."
            #     )
        else:
            self.shared_rank = 0
            self.private_rank = rank
        ####################add-4.28-end########################

        self.W_task = nn.Parameter(W0_torch.clone()) # 任务特定权重
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank)) # [256,512]  [512,rank=64]

        # self.A = torch.zeros(self.in_features, self.rank) # [768,256]
        ####################add-4.27-start######################
        self.register_buffer('A', torch.zeros(self.in_features, self.rank, device=self.W0.device)) # [768,64]
        self.register_buffer('fixed_basis', torch.empty(self.in_features, 0, device=self.W0.device))
        self.register_buffer('fixed_spectrum', torch.empty(0, device=self.W0.device))
        ####################add-4.27-end########################
        ####################add-4.28-start######################
        self.B_shared = nn.Parameter(torch.zeros(self.out_features, self.shared_rank, device=self.W0.device)) # [512, 32]：所有任务共享，持续更新
        self.B_private = nn.Parameter(torch.zeros(self.out_features, self.private_rank, device=self.W0.device)) # [512, 32]：每个任务独有，任务切换时清零
        self.register_buffer('shared_basis_block',torch.empty(self.in_features, self.shared_rank, device=self.W0.device))
        self.register_buffer('private_basis_block',torch.empty(self.in_features, self.private_rank, device=self.W0.device))
        ####################add-4.28-end########################

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
                return F.linear(x, self._compose_task_weight())
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
        if self._uses_fixed_basis(): # 策略1: 使用固定基（新增的）
            # basis_start, basis_end = self._assign_task_subspace() # # 1. 分配子空间  # 例如: Task 0 → [0:64), Task 1 → [64:128), ...
            # self.B.data.zero_() # 2. 初始化 B 为 0
            # self.B.requires_grad = True # 3. 训练 B，冻结 W_task

            ####################add-4.28-start######################
            basis_info = self._assign_task_subspace() # (0,32,32,64)
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
        self.task_id += 1
        self.current_rank += self.rank
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

    def uses_shared_core(self):
        return self._uses_shared_core()
    ####################add-4.28-end######################

    def _uses_fixed_basis(self):
        #return self.subspace_policy in {"fixed_svd_basis", "fixed_fullrank_spectrum"}
        return self.subspace_policy in {"fixed_svd_basis", "fixed_fullrank_spectrum", "fixed_svd_shared_core"}

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
        if self._uses_shared_core(): # A-> shared+privated
            shared_weight = self.B_shared @ self.shared_basis_block.T # 共享系数矩阵[512, 32] @ 共享子空间基[32, 768] = [512, 768]
            private_weight = self.B_private @ self.private_basis_block.T # 私有系数矩阵[512, 32] @ 私有子空间基[32, 768] = [512, 768]
            return self.W0 + shared_weight + private_weight # 最终权重 = 原始权重 + 共享贡献 + 私有贡献
        ####################add-4.28-end######################f
        return self.W0 + self.B @ self.A.T
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
            self.active_slice = (0, self.shared_rank, private_start, private_end)
            return self.active_slice
        ####################add-4.28-end########################
        start = self.task_id * self.rank
        end = start + self.rank


        basis_block = self.fixed_basis[:, start:end] # [768,64]   特征向量
        if self.subspace_policy == "fixed_fullrank_spectrum": #  如果使用 fullrank_spectrum，用特征值缩放
            spectrum_block = self.fixed_spectrum[start:end].unsqueeze(0) # [1,low_rank=64]
            basis_block = basis_block * spectrum_block # tezhengxiangliang * tezhegnzhi 重要的方向（大特征值）贡献更多 更接近真实的协方差矩阵结构
        self.A.copy_(basis_block) # 3. 保存到 A
        self.active_slice = (start, end) # 记录当前任务使用的子空间范围
        return start, end
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
        elif self.subspace_policy == "fixed_fullrank_spectrum": # 特征值分解
            symmetric_matrix = 0.5 * (random_matrix + random_matrix.T) # 对称化随机矩阵 1/2*(A+A^T)
            eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix) # 特征值分解
            fixed_spectrum = self._repair_spectrum(eigenvalues) # 修复特征值
            #
            self.fixed_basis = eigenvectors.to(self.W0.device)  # 正交基
            self.fixed_spectrum = fixed_spectrum.to(self.W0.device) # 修复后的特征值

        ortho_error = torch.max(
            torch.abs(
                self.fixed_basis.T @ self.fixed_basis - torch.eye(self.in_features, device=self.fixed_basis.device))
        ).item() # 正交矩阵应该满足: U^T @ U = I ortho_error 应该接近 0
        print(ortho_error)

    def _repair_spectrum(self, eigenvalues):
        fixed = eigenvalues.clone() # [768]
        near_zero_mask = fixed.abs() < self.basis_eps #  修复无限接近于零的特征值
        if near_zero_mask.any():
            signs = torch.where(fixed >= 0, torch.ones_like(fixed), -torch.ones_like(fixed))
            fixed[near_zero_mask] = signs[near_zero_mask] * self.basis_eps
        self.fixed_basis_fixes = int(near_zero_mask.sum().item())
        return fixed.to(self.W0.dtype)
    ####################add-4.27-end########################