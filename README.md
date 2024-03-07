# Verifiable Neural Network with Giza Actions

This project demonstrates how to build a verifiable neural network using Giza Actions.

Models developed using the Action SDK possess a verifiable property ensuring the integrity of the inference.

## Setup

First, connect to Giza:

```
giza users login
```

## Training

To train the model, run the following command:

```
python3 train.py
```

## Transpilation to Cairo

After training, transpile the model to Cairo:

```
giza transpile mnist_model.onnx --output-path verifiable_mnist
```

This will output a `model_id` and `version_id`.

## Deployment

Deploy the model using the following command:

```
giza deployments deploy --model-id <MODEL_ID> --version-id <VERSION_ID>
```

Replace `<MODEL_ID>` and `<VERSION_ID>` with the values obtained from the previous step. This will output a `deployment_id`.

## Prediction

To make a prediction, run:

```
python3 predict.py
```

This will output a `proof_id` (request_id).

this returns the output value and initiates a proving job to generate a Stark proof of the inference.

## Download and Verify Proof

To download and verify the proof, use the following command:

```bash
giza deployments download-proof --model-id <MODEL_ID> --version-id <VERSION_ID> --deployment-id <DEPLOYMENT_ID> --proof-id <PROOF_ID> --output-path <OUTPUT_PATH>
```

Replace `<MODEL_ID>`, `<VERSION_ID>`, `<DEPLOYMENT_ID>`, `<PROOF_ID>`, and `<OUTPUT_PATH>` with the appropriate values.

For example:

```bash
giza endpoints download-proof --model-id 422 --version-id 3 --deployment-id 89 --proof-id d5e6e6e326fa4c7e8391ff46c5c68980 --output-path proof.proof
```

This will download the proof to the specified output path.

## Liscence

This is from [Build a Verifiable Neural Network with Giza Actions](https://actions.gizatech.xyz/tutorials/build-a-verifiable-neural-network-with-giza-actions) tutorial.
