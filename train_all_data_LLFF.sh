#!/bin/bash

# python train_full_pipeline.py -s /workspace/dataset/360_v2/garden -r dn_consistency --high_poly True --export_obj True --gpu 5 --refinement_time short

# 사용할 GPU 인덱스 목록
# GPUS=(0 1 2)  # 예: GPU 0, 1, 2번을 사용한다고 가정
# GPUS=(6)  # 예: GPU 0, 1, 2번을 사용한다고 가정
# GPUS=(0 1 2 3 4 5)  # 예: GPU 0, 1, 2번을 사용한다고 가정
GPUS=(4 5)  # 예: GPU 0, 1, 2번을 사용한다고 가정

# 처리할 데이터셋 목록
# DATASETS=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
# DATASETS=("bicycle" "bonsai" "counter" "kitchen" "room" "stump")
# DATASETS=("bicycle" "bonsai" "counter" "kitchen" "room" "stump")
# DATASETS=("kitchen")
# DATASETS=("garden")

# DATASETS=("fern" "flower" "fortress" "horns" "leaves" "orchids" "trex")
DATASETS=("horns" "orchids")
# DATASETS=("fern")

num_gpus=${#GPUS[@]}

i=0
for d in "${DATASETS[@]}"; do
    # Round-robin 방식으로 GPU 할당: i번째 데이터셋은 GPUS[i % num_gpus]번 GPU 사용
    gpu=${GPUS[$((i % num_gpus))]}

    echo "Starting dataset '${d}' on GPU ${gpu}..."
    python train_full_pipeline.py \
        -s /workspace/dataset/nerf_llff_data/${d} \
        -r dn_consistency \
        --high_poly True \
        --export_obj True \
        --gpu ${gpu} \
        --refinement_time short &

    i=$((i+1))

    # 현재 진행 중인 작업 수(i)가 GPU 수 이상이 되면 하나가 끝날 때까지 대기
    # 'wait -n'은 백그라운드로 실행 중인 job 중 하나가 끝날 때까지 기다린다 (bash 4.3+ 필요)
    if (( i >= num_gpus )); then
        wait -n
    fi
done

# 남아있는 모든 백그라운드 작업 완료 대기
wait
echo "모든 작업 완료!"

CUDA_VISIBLE_DEVICES=6 python metrics.py --scene_config scene_config_LLFF.json -r density --refinement_time short
echo "density metric done!"
CUDA_VISIBLE_DEVICES=6 python metrics.py --scene_config scene_config_LLFF.json -r density --refinement_time short --use_uv_texture True
echo "density metric done!"


# rename "./output" to "./output_LLFF"
# mv ./output ./output_ensemble_24view_densification_200