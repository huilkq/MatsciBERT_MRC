# MatsciBERT_MRC

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
The following commands can be used to download the preprocessed Matscholar dataset and run our pre-trained models on Matscholar.

```bash
# Download the SciERC dataset

# Download the pre-trained models
mkdir scierc_models; cd scierc_models

cd ..

# Run the pre-trained model
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
  {
    "context": [
      "variable",
      "temperature",
      "electron",
      "paramagnetic",
      "resonance",
      "studies",
      "of",
      "the",
      "NiZn",
      "ferrite",
      "/",
      "O2Si",
      "nanocomposite"
    ],
    "query": "Any inorganic solid or alloy, any non-gaseous element.",
    "start_position": [
      8,
      11
    ],
    "end_position": [
      10,
      12
    ],
    "span_position": [
      [
        8,
        10
      ],
      [
        11,
        12
      ]
    ],
    "entity_label": "MAT"
  }

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

