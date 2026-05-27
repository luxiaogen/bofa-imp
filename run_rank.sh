#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/shell_logs

datasets=(
  imagenetr_0_20
  cifar_0_10
)

lambdas=(
  0.01
  0.05
  0.1
)

topks=(
  3
  5
)

margins=(
  0.1
)

exp_id=1

for dataset in "${datasets[@]}"; do
  for lambda in "${lambdas[@]}"; do
    for topk in "${topks[@]}"; do
      for margin in "${margins[@]}"; do
        echo "========== EXP ${exp_id}: ${dataset} | text-hard-neg | lambda=${lambda} | topk=${topk} | margin=${margin} =========="

        python main.py \
          --config "exps/${dataset}.json" \
          --subspace_policy fixed_svd_shared_core \
          --basis_alloc shared_core_private_block \
          --Kt 498 \
          --shared_rank 468 \
          --shared_importance_mode column_grad_scale \
          --importance_beta 0.999 \
          --importance_alpha 1.0 \
          --text_hard_neg_lambda "${lambda}" \
          --text_hard_neg_topk "${topk}" \
          --text_hard_neg_margin "${margin}"

        echo "========== DONE EXP ${exp_id}: ${dataset} | lambda=${lambda} | topk=${topk} | margin=${margin} =========="
        exp_id=$((exp_id + 1))
      done
    done
  done
done

##!/usr/bin/env bash
#set -euo pipefail
#
#PYTHON_BIN="${PYTHON:-python3}"
#STAMP="$(date +%F_%H-%M-%S)"
#mkdir -p logs
#
#TOTAL_TASKS="${TOTAL_TASKS:-10}"
#
#if (( $# > 0 )); then
#  CONFIGS=("$@")
#else
#  # Default order: run ImageNet-R first, then CIFAR.
#  CONFIGS=(
#    "exps/imagenetr_0_20.json"
#    "exps/cifar_0_10.json"
#  )
#fi
#
## Format: "Kt shared_rank". private_rank is computed as Kt - shared_rank.
#SWEEP=(
#  "64 32"
#  "96 48"
#  "96 64"
#  "128 64"
#  "128 96"
#  "384 352"
#  "498 468"
#)
#
#for CONFIG in "${CONFIGS[@]}"; do
#  DATASET_TAG="$(basename "${CONFIG}" .json)"
#  echo "================================================================"
#  echo "[Dataset] ${DATASET_TAG} | config=${CONFIG}"
#  echo "[Order] Running this dataset with all Kt/shared_rank settings."
#  echo "================================================================"
#
#  for item in "${SWEEP[@]}"; do
#    read -r KT SHARED_RANK <<< "${item}"
#    PRIVATE_RANK=$((KT - SHARED_RANK))
#    REQUIRED_RANK=$((SHARED_RANK + TOTAL_TASKS * PRIVATE_RANK))
#    if (( PRIVATE_RANK <= 0 || REQUIRED_RANK > 768 )); then
#      echo "[Skip][${DATASET_TAG}] invalid capacity: Kt=${KT}, shared_rank=${SHARED_RANK}, private_rank=${PRIVATE_RANK}, required_rank=${REQUIRED_RANK} > 768"
#      continue
#    fi
#
#    COMMON_ARGS=(
#      --config "${CONFIG}"
#      --subspace_policy fixed_svd_shared_core
#      --basis_alloc shared_core_private_block
#      --Kt "${KT}"
#      --shared_rank "${SHARED_RANK}"
#    )
#
#    echo "[Run][${DATASET_TAG}][Method A: baseline fixed shared/private] Kt=${KT}, shared_rank=${SHARED_RANK}, private_rank=${PRIVATE_RANK}, required_rank=${REQUIRED_RANK}/768"
#    "${PYTHON_BIN}" main.py "${COMMON_ARGS[@]}" \
#      2>&1 | tee "logs/fig1_${DATASET_TAG}_method_a_Kt${KT}_sr${SHARED_RANK}_${STAMP}.log"
#
#    echo "[Run][${DATASET_TAG}][Method B: column_grad_scale] Kt=${KT}, shared_rank=${SHARED_RANK}, private_rank=${PRIVATE_RANK}, required_rank=${REQUIRED_RANK}/768, beta=0.999, alpha=1.0"
#    "${PYTHON_BIN}" main.py "${COMMON_ARGS[@]}" \
#      --shared_importance_mode column_grad_scale \
#      --importance_beta 0.999 \
#      --importance_alpha 1.0 \
#      2>&1 | tee "logs/fig1_${DATASET_TAG}_method_b_Kt${KT}_sr${SHARED_RANK}_gradscale_${STAMP}.log"
#  done
#done
