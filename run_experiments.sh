#!/bin/bash

# 开启报错即停模式（可选）：如果中间有个实验代码崩溃了，后面的实验自动停止。如果想不管报错继续跑，就删掉这行
#!/usr/bin/env bash

# 5.12
set -e

mkdir -p logs

counter=1
for dataset in imagenetr_0_20 cifar_0_10; do
  for k in 3 5 10; do
    echo "Running experiment ${counter}: ${dataset}, topk_image_then_mix, topk=${k}, tau=0.07"

    python main.py \
      --config exps/${dataset}.json \
      --subspace_policy fixed_svd_shared_core \
      --basis_alloc shared_core_private_block \
      --Kt 498 \
      --shared_rank 468 \
      --shared_importance_mode column_grad_scale \
      --importance_beta 0.999 \
      --importance_alpha 1.0 \
      --proto_select_mode topk_pairwise_mix \
      --proto_select_topk ${k} \
      --proto_select_tau 0.07

    ((counter++))
  done
done





# 5.11
#echo "开始执行第 1 个实验 baseline-imagenet-r"
#python main.py --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0
#
#echo "开始执行第 2 个实验 baseline-cifar_0_10.json"
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0
#
#echo "开始执行第 3 个实验 type:img,λ=0.01 用文本原型相似度矩阵指导图像原型相似度矩阵"
#python main.py --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0 --center_type img --text_relation_lambda 0.01
#
#echo "开始执行第 4 个实验 type:img,λ=0.05 用文本原型相似度矩阵指导图像原型相似度矩阵"
#python main.py --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0 --center_type img --text_relation_lambda 0.05
#
#echo "开始执行第 5 个实验 type:img,λ=0.1 用文本原型相似度矩阵指导图像原型相似度矩阵"
#python main.py --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0 --center_type img --text_relation_lambda 0.1
#
#echo "开始执行第 6 个实验 type:img,λ=0.5 用文本原型相似度矩阵指导图像原型相似度矩阵"
#python main.py --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0 --center_type img --text_relation_lambda 0.5
#
#echo "开始执行第 7 个实验 type:img,λ=1.0 用文本原型相似度矩阵指导图像原型相似度矩阵"
#python main.py --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0 --center_type img --text_relation_lambda 1.0

#echo "开始执行第 1 个实验 topk=20,λ=0.1..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 0.1 --shared_svd_reg_topk 20
#echo "开始执行第 2 个实验 topk=20,λ=0.5..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 0.5 --shared_svd_reg_topk 20
#echo "开始执行第 3 个实验 topk=20,λ=1..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 1.0 --shared_svd_reg_topk 20
#echo "开始执行第 4 个实验 topk=20,λ=5..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 5.0 --shared_svd_reg_topk 20
#echo "开始执行第 5 个实验 topk=20,λ=100..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 100.0 --shared_svd_reg_topk 20
#echo "cifar100所有实验跑完了....."
#echo "开始执行第 -r 个实验 topk=20,λ=0.1..."
#python main.py --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 0.1 --shared_svd_reg_topk 20


#echo "开始执行第 1 个实验..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 0.5 --shared_svd_reg_topk 468
#
#echo "开始执行第 2 个实验..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 1 --shared_svd_reg_topk 468
#
#echo "开始执行第 3 个实验..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 0.5 --shared_svd_reg_topk 150
#
#echo "开始执行第 4 个实验..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 0.1 --shared_svd_reg_topk 150
#
#echo "开始执行第 5 个实验..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 1 --shared_svd_reg_topk 150
#
#echo "开始执行第 6 个实验..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 5 --shared_svd_reg_topk 150
#
#echo "开始执行第 7 个实验..."
#python main.py --config exps/cifar_0_10.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_svd_reg_lambda 100 --shared_svd_reg_topk 150
#
#echo "所有实验跑完啦！"
