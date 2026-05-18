#!/usr/bin/env bash
set -e

mkdir -p logs/shell_logs

DATASETS=("imagenetr_0_20" "cifar_0_10")

ROUTE_TOPM=("3" "5" "10")
ROUTE_TAU=("2.0" "5.0" "5.0")

EMA_BETAS=("0.2" "0.3" "0.5" "0.9")

exp_id=1

for dataset in "${DATASETS[@]}"; do
  echo "========== Dataset: ${dataset} =========="

  BASE_ARGS="
    --config exps/${dataset}.json
    --subspace_policy fixed_svd_shared_core
    --basis_alloc shared_core_private_block
    --Kt 498
    --shared_rank 468
  "

  echo "========== Phase 1: private routing only =========="

  for i in "${!ROUTE_TOPM[@]}"; do
    topm="${ROUTE_TOPM[$i]}"
    tau="${ROUTE_TAU[$i]}"

    echo "========== EXP ${exp_id}: ${dataset} | route-only | topm=${topm} | tau=${tau} =========="

    python main.py ${BASE_ARGS} \
      --private_route_mode task_topm \
      --private_route_topm "${topm}" \
      --private_route_tau "${tau}"

    ((exp_id++))
  done

  echo "========== Phase 2: private routing + shared EMA =========="

  for beta in "${EMA_BETAS[@]}"; do
    for i in "${!ROUTE_TOPM[@]}"; do
      topm="${ROUTE_TOPM[$i]}"
      tau="${ROUTE_TAU[$i]}"

      echo "========== EXP ${exp_id}: ${dataset} | route+ema | topm=${topm} | tau=${tau} | ema_beta=${beta} =========="

      python main.py ${BASE_ARGS} \
        --private_route_mode task_topm \
        --private_route_topm "${topm}" \
        --private_route_tau "${tau}" \
        --shared_ema_mode task_end \
        --shared_ema_beta "${beta}"

      ((exp_id++))
    done
  done
done