import csv
import json


#
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


def bio_to_query(text, label):
    # 构造实体列表和对应的文本片段
    entities = []
    entity_start = None
    prev_label = None
    for i, (word, tag) in enumerate(zip(text, label)):
        if tag.startswith('B-'):
            if prev_label is not None:
                entities.append((prev_label[2:], text[entity_start:i]))
            entity_start = i
        elif tag.startswith('I-'):
            if prev_label is None or prev_label[2:] != tag[2:]:
                entity_start = i
        prev_label = tag
    if prev_label is not None:
        entities.append((prev_label[2:], text[entity_start:]))

    # 构造查询字符串
    query = ""
    for entity_type, entity_text in entities:
        if entity_type in ["DNA", "RNA", "cell_line", "cell_type", "protein"]:
            query += f"Can you detect {entity_type} entities like：{entity_text}\n"
            # print(query)
            # 将MRC数据集写入txt文件
            with open('D:\pythonProject\MatsciBERT_MRC\datasets\JNLPBA\\query_train.txt', 'w', encoding='utf-8') as f:
                f.write(f"{query} \n")
    return query


# 读取tsv数据集
# ner_data = read_tsv('D:\pythonProject\MatsciBERT_MRC\datasets\JNLPBA\\train.tsv')
#
# # 读取tsv文件
# # 构造MRC数据集
# mrc_data = []
# for data in ner_data:
#     text = data["words"]
#     entities = data["labels"]
#     # print(text)
#     bio_to_query(text, entities)

# 将MRC数据集写入txt文件
# with open('D:\pythonProject\MatsciBERT_MRC\datasets\JNLPBA\\query_train..txt', 'w', encoding='utf-8') as f:
#     for data in mrc_data:
#         f.write(f"{data['context']} {data['question']} {data['answer']}\n")


import json

# 定义每种实体类型对应的query
queries = {
    "DNA": "deoxyribonucleic acid",
    "RNA": "ribonucleic acid",
    "cell_line": "cell line",
    "cell_type": "cell type",
    "protein": "protein entities are limited to nitrogenous organic compounds and are parts of all living organisms, as structural components of body tissues such as muscle, hair, collagen and as enzymes and antibodies."
}

# 定义函数将BIO格式数据集转换成query-BIO格式
def bio_to_query_bios(bio_file_path, query_bio_file_path):
    data = read_tsv('D:\pythonProject\MatsciBERT_MRC\datasets\JNLPBA\\train.tsv')

    query_bio_data = []
    for instance in data:
        sentence = instance["words"]
        labels = instance["labels"]

        # 初始化query-BIO标记
        query_bio_labels = ["O"] * len(labels)
        print(query_bio_labels)
        # 遍历每个实体，并在其位置上加入query-BIO标记
        for label in labels:
            entity_type = label[2:]
            if entity_type in queries:
                start_index = label[0]
                end_index = label[1]
                query = queries[entity_type]

                query_bio_labels[start_index] = "B-" + query
                for i in range(start_index + 1, end_index):
                    query_bio_labels[i] = "I-" + query

        # 将句子和query-BIO标记保存到新的数据集中
        new_instance = {"text": sentence, "labels": query_bio_labels}
        query_bio_data.append(new_instance)

    # 将新生成的query-BIO格式的数据集保存到文件中
    with open(query_bio_file_path, "w", encoding="utf-8") as f:
        json.dump(query_bio_data, f, ensure_ascii=False)


bio_to_query_bios('D:\pythonProject\MatsciBERT_MRC\datasets\JNLPBA\\train.tsv','D:\pythonProject\MatsciBERT_MRC\datasets\JNLPBA\\qurey_train.json')