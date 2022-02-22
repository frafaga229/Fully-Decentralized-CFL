cd ../../

echo "Experiment for synthetic data"
cd data/synthetic/ || exit

echo "Generate synthetic data"
python generate_data.py \
    --n_clients 300 \
    --dimension 150 \
    --n_clusters 2 \
    --graph_hetero_level 0.2 \
    --data_hetero_level 1.0 \
    --noise_level 0.0 \
    --n_train 1000 \
    --n_val 600 \
    --n_test 600 \
    --seed 12345

cd ../..

echo "Train local model (no communication)"
python run.py \
    synthetic \
    local \
    --input_dimension 150 \
    --n_rounds 200 \
    --bz 256 \
    --local_steps 1 \
    --log_freq 10 \
    --device cpu \
    --optimizer sgd \
    --lr 0.1 \
    --lr_scheduler constant \
    --verbose 1 \
    --logs_dir "logs/local" \
    --seed 1234

echo "Train global model (FedAvg)"
python run.py \
    synthetic \
    FedAvg \
    --input_dimension 150 \
    --n_rounds 200 \
    --bz 256 \
    --local_steps 1 \
    --log_freq 10 \
    --device cpu \
    --optimizer sgd \
    --lr 0.1 \
    --lr_scheduler constant \
    --verbose 1 \
    --logs_dir "logs/FedAvg" \
    --seed 1234

echo "Train Clustered FL"
python run.py \
    synthetic \
    clustered \
    --input_dimension 150 \
    --n_rounds 200 \
    --bz 256 \
    --local_steps 1 \
    --log_freq 10 \
    --device cpu \
    --optimizer sgd \
    --lr 0.1 \
    --lr_scheduler constant \
    --verbose 1 \
    --logs_dir "logs/clustered" \
    --seed 1234

echo "Make plots"
python make_plots --logs_dir logs save_dir results
echo "PLots save to results/"
