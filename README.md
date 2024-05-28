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

## Entity Model

### Input data format for the entity model
```
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
```
### Train/evaluate the entity model

You can use `run_entity.py` with `--do_train` to train an entity model and with `--do_eval` to evaluate an entity model.
A trianing command template is as follow:
```bash
python run_ner_bert_mac.py \
    --do_train --do_eval [--eval_test] \
    --learning_rate=2e-5 \
    --max_seq_length=512 \
    --other_learning_rate=1e-3 \
    --train_batch_size=8 \
    --data_dir {directory of preprocessed dataset} \
    --model_name_or_path {bert-base-uncased | matscibert | scibert | biobert} \
    --output_dir {directory of output files} \
    --loss_type {lsr | focal | ce}
```
Arguments:
* `--learning_rate`: the learning rate for BERT encoder parameters.
* `--other_learning_rate`: the learning rate for task-specific parameters, i.e., the classifier head after the encoder.
* `--loss_type`: the loss_type used in the model. 
* `--model_name_or_path`: the base transformer model. 
* `--eval_test`: whether evaluate on the test set or not.

Put train, dev, test and other data in the datasets folder (sample data has been given, as long as it conforms to the format)

