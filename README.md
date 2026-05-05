# sh

## 固定正交基，每个任务（特定）相同的 rank=64个方向
```shell
python --config exps/imagenetr_0_20.json \
       --subspace_policy fixed_svd_basis \
       --Kt 64
```

## 固定正交基，每个任务（特定）相同的 rank=64个方向+一个共享的方向 rank=32
```shell
python main.py \
  --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_shared_core \
  --Kt 64 --shared_rank 32 \
  --shared_lr_scale 0.1 \
  --basis_alloc \
  shared_core_private_block
```

## 首任务大 rank（特定），后续任务平分 rank=30
```shell
python main.py \
  --config exps/imagenetr_0_20.json \
  --subspace_policy fixed_svd_basis \
  --basis_alloc front_loaded_block \
  --Kt 30
  
 sleep 7200 &&python main.py \
  --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_basis \
  --basis_alloc front_loaded_block \
  --Kt 30
```

## 首任务大 rank（共享），后续任务平分 rank=30
```shell
sleep 7200 &&python main.py \
  --config exps/imagenetr_0_20.json \
  --subspace_policy fixed_svd_shared_core \
  --basis_alloc shared_core_private_block \
  --Kt 498 \
  --shared_rank 468

sleep 7200 &&python main.py \
  --config exps/cifar_0_10.json \
  --subspace_policy fixed_svd_shared_core \
  --basis_alloc shared_core_private_block \
  --Kt 498 \
  --shared_rank 468
```