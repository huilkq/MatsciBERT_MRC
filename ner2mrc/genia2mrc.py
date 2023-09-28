#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
# file: genia2mrc.py

import os
import json

import pandas as pd


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for index, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
            chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = index

            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    lines = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        words = []
        labels = []
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        for line in reader:
            if len(line) == 0 or line == "\n":
                if words:
                    lines.append({"words": words, "labels": labels})
                    words = []
                    labels = []
            else:
                words.append(line[0])
                if len(line) > 1:
                    labels.append(line[1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"words": words, "labels": labels})
        return lines


def read_text(input_file):
    lines = []
    with open(input_file, 'r', encoding="utf-8-sig") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": labels})
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"words": words, "labels": labels})
    return lines

def read_json(input_file):
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            labels = line['ner_tags']
            words = line['tokens']
            lines.append({"words": words, "labels": labels})
    return lines

import json

# 定义每种实体类型对应的query
# queries_Metal = {'MAT': 'Any inorganic solid or alloy, any non-gaseous element.',
#                  'SPL': 'Names for crystal structures/phases.',
#                  'DSC': 'Special descriptions of the type/shape of the sample.',
#                  'PRO': 'Anything measurable that can have a unit and a value.',
#                  'APL': 'Any high-level application such as photovoltaics, or any specific device such as field-effect transistor.',
#                  'CMT': 'Any technique for synthesising a material.',
#                  'SMT': 'Any method used to characterize a material.'}

queries_Metal = {'MAT': 'Materials made from inorganic substances alone or in combination with other substances.',
                 'SPL': 'Symmetry/phase label.',
                 'DSC': 'Sample descriptor.',
                 'PRO': 'In science, a property is anything that describes a material or substance..',
                 'APL': 'Application is a purpose that material or object can be used for..',
                 'CMT': 'Synthesize is to make new compounds from simpler elements..',
                 'SMT': 'The structural characteristics of materials are revealed and determined by various physical and chemical testing methods.'}

queries_SOFC = {'MATERIAL': 'Fuel cell’s material.',
                'DEVICE': 'The type of device used in the fuel cell experiment (e.g., IT-SOFC).',
                'EXPERIMENT': 'Experiment related operations such as test, perform, and report.',
                'VALUE': 'Annotate the values and their respective units with the VALUE type (for example, above 750◦C, 1.0 W cm −2).'}

queries_SOFC_slot = {'anode_material': 'Fuel cell’s anode.',
                     'cathode_material': 'Fuel cell’s cathode.',
                     'conductivity': 'A measure of a substance’s ability to carry current.',
                     'current_density': 'Current per unit section area.',
                     'degradation_rate': 'Rate of material consumption.',
                     'device': 'The type of device used in the fuel cell experiment (e.g., IT-SOFC).',
                     'electrolyte_material': 'Fuel cell’s electrolyte material.',
                     'fuel_used': 'The chemical composition or the class of a fuel or the oxidant species.',
                     'interlayer_material': 'Fuel cell’s interlayer material.',
                     'open_circuit_voltage': 'The potential difference between the positive and negative electrodes of a battery when no current passes through.',
                     'power_density': 'The amount of power processed per unit volume.',
                     'resistance': 'The more the conductor impedes the current.',
                     'support_material': 'Fuel cell’s support material.',
                     'time_of_operation': 'The time it takes to run.',
                     'thickness': 'The distance between the upper and lower sides of an object.',
                     'voltage': 'A physical quantity that measures the difference in energy of a unit charge in an electrostatic field due to the difference in electric potential.',
                     'working_temperature': 'Temperature class.'}


queries_CHEMD = {'Chemical': 'Pure substances and mixtures composed of various elements.'}


# 定义函数将BIO格式数据集转换为（上下文，查询，答案）三元组
def bio_to_triples(bio_file_path, triples_file_path):
    data = read_text(bio_file_path)
    origin_count = 0
    new_count = 0
    triples_data = []
    for instance in data:
        origin_count += 1
        sentence = instance["words"]
        labels = instance["labels"]
        labels = get_entities(labels, id2label=None, markup='bio')
        print(labels)
        # 遍历每个实体，并构造（上下文，查询，答案）三元组
        for tag_idx, (tag, query) in enumerate(queries_Metal.items()):
            positions = []
            for label in labels:
                if label[0] == tag:
                    positions.append([label[1], label[2] + 1])
            print(positions)
            # for labe; for (position)
            mrc_sample = {
                "context": sentence,
                "query": query,
                "start_position": [x[0] for x in positions],
                "end_position": [x[1] for x in positions],
                "span_position": positions,
                "entity_label": tag,
                "impossible": "false",
                "qas_id": f"{origin_count}.{tag_idx}"
            }
            triples_data.append(mrc_sample)
            new_count += 1

    # 将新生成的（上下文，查询，答案）三元组保存到文件中
    json.dump(triples_data, open(triples_file_path, "w", encoding="utf-8-sig"), ensure_ascii=False, indent=2)


def main():
    genia_raw_dir = r"D:\pythonProject\MatsciBERT_MRC\datasets\Metal"
    genia_mrc_dir = r"D:\pythonProject\MatsciBERT_MRC\datasets\Metal\mrc_format"
    tag2query_file = "queries/Matscholar.json"
    os.makedirs(genia_mrc_dir, exist_ok=True)
    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(genia_raw_dir, f"{phase}.txt")
        new_file = os.path.join(genia_mrc_dir, f"query_{phase}.json")
        bio_to_triples(old_file, new_file)


if __name__ == '__main__':
    main()
