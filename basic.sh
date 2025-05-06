#!/bin/bash

if [ -z "$1" ]; then
  echo "Entry-point: data-preprocessing, hyperparameters-tuning, model-training, data-evaluation"
  exit 1
fi

ENTRY=$1

case "$ENTRY" in
  data-preprocessing)
    mlflow run . \
      --entry-point data-preprocessing \
      --env-manager local \
      --experiment-name Lions_vs_Leopards \
      --run-name Data_Preprocessing \
      -P raw-data-dir=raw-images \
      -P output-dir=dataset \
      -P val-size=0.2
    ;;
  
  hyperparameters-tuning)
    mlflow run . \
      --entry-point hyperparameters-tuning \
      --env-manager local \
      --experiment-name Lions_vs_Leopards \
      --run-name Hyperparameters_Search \
      -P n-trials=10
    ;;

  model-training)
    mlflow run . \
      --entry-point model-training \
      --env-manager local \
      --experiment-name Lions_vs_Leopards \
      --run-name Model_Training \
      -P epochs=3 \
      -P batch-size=32 \
      -P learning-rate=0.001
    ;;

  data-evaluation)
    mlflow run . \
      --entry-point data-evaluation \
      --env-manager local \
      --experiment-name Lions_vs_Leopards \
      --run-name Evaluation \
      -P eval-dataset=dataset/val
    ;;

  *)
    echo "Неизвестный entry-point: $ENTRY"
    exit 1
    ;;
esac
