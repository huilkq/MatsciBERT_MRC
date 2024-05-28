# MatsciBERT_MRC
## Overview
![](./figs/overview.png)
In this work, we present a simple approach for entity and relation extraction. Our approach contains three conponents:

1. The **entity model** takes a piece of text as input and predicts all the entities at once.
2. The **relation model** considers every pair of entities independently by inserting typed entity markers, and predicts the relation type for each pair.
3. The **approximation relation model** supports batch computations, which enables efficient inference for the relation model.

Please find more details of this work in our [paper](https://arxiv.org/pdf/2010.12812.pdf).

## Setup

### Install dependencies
```
Python==3.8.12; 
torch==1.12.1;
numpy;
pandas
```

### Download and preprocess the datasets
Our experiments are based on five datasets: BC4CHEMD, Matscholar, NLMChem, SOFC-Slot and SOFC. Please find the links and pre-processing below:
* ACE04/ACE05: We use the preprocessing code from [DyGIE repo](https://github.com/luanyi/DyGIE/tree/master/preprocessing). Please follow the instructions to preprocess the ACE05 and ACE04 datasets.
* SciERC: The preprocessed SciERC dataset can be downloaded in their project [website](http://nlp.cs.washington.edu/sciIE/).

## Quick Start
The following commands can be used to download the preprocessed SciERC dataset and run our pre-trained models on SciERC.

```bash
# Download the SciERC dataset
wget http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
mkdir scierc_data; tar -xf sciERC_processed.tar.gz -C scierc_data; rm -f sciERC_processed.tar.gz
scierc_dataset=scierc_data/processed_data/json/

# Download the pre-trained models (single-sentence)
mkdir scierc_models; cd scierc_models

# Download the pre-trained entity model
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx0.zip
unzip ent-scib-ctx0.zip; rm -f ent-scib-ctx0.zip
scierc_ent_model=scierc_models/ent-scib-ctx0/

# Download the pre-trained full relation model
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx0.zip
unzip rel-scib-ctx0.zip; rm -f rel-scib-ctx0.zip
scierc_rel_model=scierc_models/rel-scib-ctx0/

# Download the pre-trained approximation relation model
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel_approx-scib-ctx0.zip
unzip rel_approx-scib-ctx0.zip; rm -f rel_approx-scib-ctx0.zip
scierc_rel_model_approx=scierc_models/rel_approx-scib-ctx0/

cd ..

# Run the pre-trained entity model, the result will be stored in ${scierc_ent_model}/ent_pred_test.json
python run_entity.py \
    --do_eval --eval_test \
    --context_window 0 \
    --task scierc \
    --data_dir ${scierc_dataset} \
    --model allenai/scibert_scivocab_uncased \
    --output_dir ${scierc_ent_model}



## Entity Model

### Input data format for the entity model

The input data format of the entity model is JSONL. Each line of the input file contains one document in the following format.

### Train/evaluate the entity model

You can use `run_entity.py` with `--do_train` to train an entity model and with `--do_eval` to evaluate an entity model.
A trianing command template is as follow:
```bash
python run_entity.py \
    --do_train --do_eval [--eval_test] \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window {0 | 100 | 300} \
    --task {ace05 | ace04 | scierc} \
    --data_dir {directory of preprocessed dataset} \
    --model {bert-base-uncased | albert-xxlarge-v1 | allenai/scibert_scivocab_uncased} \
    --output_dir {directory of output files}
```
Arguments:
* `--learning_rate`: the learning rate for BERT encoder parameters.
* `--task_learning_rate`: the learning rate for task-specific parameters, i.e., the classifier head after the encoder.
* `--context_window`: the context window size used in the model. `0` means using no contexts. In our cross-sentence entity experiments, we use `--context_window 300` for BERT models and SciBERT models and use `--context_window 100` for ALBERT models.
* `--model`: the base transformer model. We use `bert-base-uncased` and `albert-xxlarge-v1` for ACE04/ACE05 and use `allenai/scibert_scivocab_uncased` for SciERC.
* `--eval_test`: whether evaluate on the test set or not.

The predictions of the entity model will be saved as a file (`ent_pred_dev.json`) in the `output_dir` directory. If you set `--eval_test`, the predictions (`ent_pred_test.json`) are on the test set. The prediction file of the entity model will be the input file of the relation model.



*I. Data & Model:
**Put train, dev, test and other data in the datasets folder (sample data has been given, as long as it conforms to the format)

