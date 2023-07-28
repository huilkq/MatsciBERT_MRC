import json


def _read_json(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    # input_data = input_data[:10]

    examples = []
    for entry in input_data:
        query_item = entry["query"]
        context_item = entry["context"]
        span_position = entry["span_position"]
        query_tokens = query_item.split()
        for i, token in enumerate(query_tokens):
            context_item.append(token)
        examples.append({"words": context_item, "labels": span_position})
    print(len(examples))
    return examples


lines = _read_json("D:\pythonProject\MatsciBERT_MRC\datasets\Metal\mrc_format\query_train.json")
print(lines[:10])
for (i, line) in enumerate(lines):
    # if i == 0:
    # continue
    # guid = "%s-%s" % (set_type, i + 1)
    text_a = line['words']
    labels = line['labels']
    print(labels)