#!/usr/bin/env bash
# bash run_experiments.sh 2>&1 | tee logs/ema_update_$(date +%F_%H-%M-%S).log
#echo "exp 1 首任务秩长一点（共享），后面均分（特定） baseline imagenet-r"
#python main.py \
#	--config exps/imagenetr_0_20.json \
#	--subspace_policy fixed_svd_shared_core \
#	--basis_alloc shared_core_private_block \
#	--Kt 498 --shared_rank 468

echo "exp 1 W_shared_ema 0.1"

python main.py --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_shared_core \
  --basis_alloc shared_core_private_block \
  --Kt 498 \
  --shared_rank 468 \
  --shared_importance_mode column_grad_scale \
	--importance_beta 0.999 \
	--importance_alpha 1.0 \
  --shared_ema_mode task_end \
  --shared_ema_beta 0.1


echo "exp 2 W_shared_ema 0.2"

python main.py --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_shared_core \
  --basis_alloc shared_core_private_block \
  --Kt 498 \
  --shared_rank 468 \
  --shared_importance_mode column_grad_scale \
	--importance_beta 0.999 \
	--importance_alpha 1.0 \
  --shared_ema_mode task_end \
  --shared_ema_beta 0.2

echo "exp 3 W_shared_ema 0.3"

python main.py --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_shared_core \
  --basis_alloc shared_core_private_block \
  --Kt 498 \
  --shared_rank 468 \
  --shared_importance_mode column_grad_scale \
	--importance_beta 0.999 \
	--importance_alpha 1.0 \
  --shared_ema_mode task_end \
  --shared_ema_beta 0.3

echo "exp 4 W_shared_ema 0.5"

python main.py --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_shared_core \
  --basis_alloc shared_core_private_block \
  --Kt 498 \
  --shared_rank 468 \
  --shared_importance_mode column_grad_scale \
	--importance_beta 0.999 \
	--importance_alpha 1.0 \
  --shared_ema_mode task_end \
  --shared_ema_beta 0.5

echo "exp 5 W_shared_ema 0.9"

python main.py --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_shared_core \
  --basis_alloc shared_core_private_block \
  --Kt 498 \
  --shared_rank 468 \
  --shared_importance_mode column_grad_scale \
	--importance_beta 0.999 \
	--importance_alpha 1.0 \
  --shared_ema_mode task_end \
  --shared_ema_beta 0.9



#echo "exp 1 W_shared_ema 0.3"
#
#python main.py --config exps/cifar_0_10.json \
#  --subspace_policy fixed_svd_shared_core \
#  --basis_alloc shared_core_private_block \
#  --Kt 498 \
#  --shared_rank 468 \
#  --shared_ema_mode task_end \
#  --shared_ema_beta 0.2

#echo "exp 2 W_shared_ema 0.5"
#
#python main.py --config exps/cifar_0_10.json \
#  --subspace_policy fixed_svd_shared_core \
#  --basis_alloc shared_core_private_block \
#  --Kt 498 \
#  --shared_rank 468 \
#  --shared_ema_mode task_end \
#  --shared_ema_beta 0.5
#
#  echo "exp 3 W_shared_ema 0.9"
#
#python main.py --config exps/cifar_0_10.json \
#  --subspace_policy fixed_svd_shared_core \
#  --basis_alloc shared_core_private_block \
#  --Kt 498 \
#  --shared_rank 468 \
#  --shared_ema_mode task_end \
#  --shared_ema_beta 0.9
#mkdir -p logs
#
#COMMON_ARGS=(
#  --subspace_policy fixed_svd_shared_core
#  --basis_alloc shared_core_private_block
#  --Kt 498
#  --shared_rank 468
#  --shared_importance_mode column_grad_scale
#  --importance_beta 0.999
#  --importance_alpha 1.0
#)
#
#run_baseline() {
#  local dataset="$1"
#
#  echo "Running baseline: ${dataset}, private_route=none"
#  python main.py \
#    --config "exps/${dataset}.json" \
#    "${COMMON_ARGS[@]}"
#}
#
#run_route() {
#  local dataset="$1"
#  local topm="$2"
#  local tau="$3"
#
#  echo "Running route: ${dataset}, private_route=task_topm, topm=${topm}, tau=${tau}"
#  python main.py \
#    --config "exps/${dataset}.json" \
#    "${COMMON_ARGS[@]}" \
#    --private_route_mode task_topm \
#    --private_route_topm "${topm}" \
#    --private_route_tau "${tau}"
#}
#
#for dataset in imagenetr_0_20 cifar_0_10; do
#  run_baseline "${dataset}"
#
#  run_route "${dataset}" 3 2.0
#  run_route "${dataset}" 5 5.0
#  run_route "${dataset}" 10 5.0
#done
#
#echo "All routing experiments finished."


##!/bin/bash
#
## 开启报错即停模式（可选）：如果中间有个实验代码崩溃了，后面的实验自动停止。如果想不管报错继续跑，就删掉这行
##!/usr/bin/env bash
#
## 5.12
#set -e
#
#mkdir -p logs
#echo "running exp 0  测试下 -r 的准确度有没有下降"
#python --config exps/imagenetr_0_20.json --subspace_policy fixed_svd_shared_core --basis_alloc shared_core_private_block --Kt 498 --shared_rank 468 --shared_importance_mode column_grad_scale --importance_beta 0.999 --importance_alpha 1.0
#
#counter=1
#for dataset in imagenetr_0_20 cifar_0_10; do
#  for k in 3 5 10; do
#    echo "Running experiment ${counter}: ${dataset}, topk_image_then_mix, topk=${k}, tau=0.07"
#
#    python main.py \
#      --config exps/${dataset}.json \
#      --subspace_policy fixed_svd_shared_core \
#      --basis_alloc shared_core_private_block \
#      --Kt 498 \
#      --shared_rank 468 \
#      --shared_importance_mode column_grad_scale \
#      --importance_beta 0.999 \
#      --importance_alpha 1.0 \
#      --proto_select_mode topk_pairwise_mix \
#      --proto_select_topk ${k} \
#      --proto_select_tau 0.07
#
#    ((counter++))
#  done
#done





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
