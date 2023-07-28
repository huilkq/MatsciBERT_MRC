""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """
import torch
import logging
import os
import copy
import json

from transformers import BasicTokenizer

from .utils_ner import DataProcessor, get_entities

logger = logging.getLogger(__name__)
basicTokenizer = BasicTokenizer()


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, query_item, context_item, start_position, end_position, entity_label):
        self.guid = guid
        self.query_item = query_item
        self.context_item = context_item
        self.start_position = start_position
        self.end_position = end_position
        self.entity_label = entity_label

    # def __init__(self, guid, text_a, subject):
    #     self.guid = guid
    #     self.text_a = text_a
    #     self.subject = subject

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, start_ids, end_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_start_ids = all_start_ids[:, :max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_lens


# def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
#                                  cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
#                                  sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
#                                  sequence_a_segment_id=0, mask_padding_with_zero=True, ):
#     """ Loads a data file into a list of `InputBatch`s
#         `cls_token_at_end` define the location of the CLS token:
#             - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
#             - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
#         `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
#     """
#     label2id = {label: i for i, label in enumerate(label_list)}
#     print(label2id)
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d", ex_index, len(examples))
#         textlist = example.text_a
#         subjects = example.subject
#         # print(subjects)
#         if isinstance(textlist, list):
#             textlist = " ".join(textlist)
#         tokens = tokenizer.tokenize(textlist)
#         start_ids = [0] * len(tokens)
#         end_ids = [0] * len(tokens)
#         subjects_id = []
#         for subject in subjects:
#             label = subject[0]
#             start = subject[1]
#             end = subject[2]
#             start_ids[start] = label2id[label]
#             end_ids[end] = label2id[label]
#             subjects_id.append((label2id[label], start, end))
#         # Account for [CLS] and [SEP] with "- 2".
#         special_tokens_count = 2
#         if len(tokens) > max_seq_length - special_tokens_count:
#             tokens = tokens[: (max_seq_length - special_tokens_count)]
#             start_ids = start_ids[: (max_seq_length - special_tokens_count)]
#             end_ids = end_ids[: (max_seq_length - special_tokens_count)]
#         # The convention in BERT is:
#         # (a) For sequence pairs:
#         #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#         #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
#         # (b) For single sequences:
#         #  tokens:   [CLS] the dog is hairy . [SEP]
#         #  type_ids:   0   0   0   0  0     0   0
#         #
#         # Where "type_ids" are used to indicate whether this is the first
#         # sequence or the second sequence. The embedding vectors for `type=0` and
#         # `type=1` were learned during pre-training and are added to the wordpiece
#         # embedding vector (and position vector). This is not *strictly* necessary
#         # since the [SEP] token unambiguously separates the sequences, but it makes
#         # it easier for the model to learn the concept of sequences.
#         #
#         # For classification tasks, the first vector (corresponding to [CLS]) is
#         # used as as the "sentence vector". Note that this only makes sense because
#         # the entire model is fine-tuned.
#         tokens += [sep_token]
#         start_ids += [0]
#         end_ids += [0]
#         segment_ids = [sequence_a_segment_id] * len(tokens)
#         if cls_token_at_end:
#             tokens += [cls_token]
#             start_ids += [0]
#             end_ids += [0]
#             segment_ids += [cls_token_segment_id]
#         else:
#             tokens = [cls_token] + tokens
#             start_ids = [0] + start_ids
#             end_ids = [0] + end_ids
#             segment_ids = [cls_token_segment_id] + segment_ids
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#         input_len = len(input_ids)
#         # Zero-pad up to the sequence length.
#         padding_length = max_seq_length - len(input_ids)
#         if pad_on_left:
#             input_ids = ([pad_token] * padding_length) + input_ids
#             input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
#             segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
#             start_ids = ([0] * padding_length) + start_ids
#             end_ids = ([0] * padding_length) + end_ids
#         else:
#             input_ids += [pad_token] * padding_length
#             input_mask += [0 if mask_padding_with_zero else 1] * padding_length
#             segment_ids += [pad_token_segment_id] * padding_length
#             start_ids += ([0] * padding_length)
#             end_ids += ([0] * padding_length)
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#         assert len(start_ids) == max_seq_length
#         assert len(end_ids) == max_seq_length
#
#         if ex_index < 5:
#             logger.info("*** Example ***")
#             logger.info("guid: %s", example.guid)
#             logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
#             logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
#             logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
#             logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
#             logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))
#
#         features.append(InputFeature(input_ids=input_ids,
#                                      input_mask=input_mask,
#                                      segment_ids=segment_ids,
#                                      start_ids=start_ids,
#                                      end_ids=end_ids,
#                                      subjects=subjects_id,
#                                      input_len=input_len))
#     return features


def convert_examples_to_features(examples, tokenizer, label_list, max_seq_length, is_training=True,
                                 allow_impossible=True, pad_sign=True):
    label_map = {tmp: idx for idx, tmp in enumerate(label_list)}
    features = []
    for (example_idx, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.query_item)
        # whitespace_doc = whitespace_tokenize(example.context_item)
        textlist = example.context_item
        # print(query_tokens)
        if isinstance(textlist, list):
            textlist = " ".join(textlist)
        whitespace_doc = tokenizer.tokenize(textlist)
        # print(whitespace_doc)
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        if len(example.start_position) == 0 and len(example.end_position) == 0:
            doc_start_pos = []
            doc_end_pos = []
            all_doc_tokens = []

            for token_item in example.context_item:
                tmp_subword_lst = tokenizer.tokenize(token_item)
                all_doc_tokens.extend(tmp_subword_lst)
            doc_start_pos = [0] * len(all_doc_tokens)
            doc_end_pos = [0] * len(all_doc_tokens)
            # doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

        else:
            doc_start_pos = []
            doc_end_pos = []
            # doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

            all_doc_tokens = []
            offset_idx_dict = {}

            fake_start_pos = [0] * len(whitespace_doc)
            fake_end_pos = [0] * len(whitespace_doc)

            for start_item in example.start_position:
                fake_start_pos[int(start_item)] = 1
            for end_item in example.end_position:
                # if int(end_item) >= len(fake_end_pos):
                #     print("stop")
                fake_end_pos[int(end_item) - 1] = 1

            # improve answer span
            for idx, (token, start_label, end_label) in enumerate(
                    zip(example.context_item, fake_start_pos, fake_end_pos)):
                tmp_subword_lst = tokenizer.tokenize(token)
                # print(tmp_subword_lst)
                if len(tmp_subword_lst) > 1:
                    offset_idx_dict[idx] = len(all_doc_tokens)
                    doc_start_pos.append(start_label)
                    doc_start_pos.extend([0] * (len(tmp_subword_lst) - 1))

                    doc_end_pos.extend([0] * (len(tmp_subword_lst) - 1))
                    doc_end_pos.append(end_label)

                    all_doc_tokens.extend(tmp_subword_lst)
                elif len(tmp_subword_lst) == 1:
                    offset_idx_dict[idx] = len(all_doc_tokens)
                    doc_start_pos.append(start_label)
                    doc_end_pos.append(end_label)
                    all_doc_tokens.extend(tmp_subword_lst)
        assert len(all_doc_tokens) == len(doc_start_pos)
        assert len(all_doc_tokens) == len(doc_end_pos)
        assert len(doc_start_pos) == len(doc_end_pos)

        if len(all_doc_tokens) >= max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]
            doc_start_pos = doc_start_pos[: max_tokens_for_doc]
            doc_end_pos = doc_end_pos[: max_tokens_for_doc]

        input_tokens = []
        segment_ids = []
        input_mask = []
        start_pos = []
        end_pos = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)

        for query_item in query_tokens:
            input_tokens.append(query_item)
            segment_ids.append(0)
            start_pos.append(0)
            end_pos.append(0)

        input_tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        start_pos.append(0)
        end_pos.append(0)

        input_tokens.extend(all_doc_tokens)
        segment_ids.extend([1] * len(all_doc_tokens))
        start_pos.extend(doc_start_pos)
        end_pos.extend(doc_end_pos)

        input_tokens.append("[SEP]")
        segment_ids.append(1)
        start_pos.append(0)
        end_pos.append(0)
        input_mask = [1] * len(input_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_len = len(input_ids)
        # zero-padding up to the sequence length
        if len(input_ids) < max_seq_length and pad_sign:
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            start_pos += padding
            end_pos += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_pos) == max_seq_length
        assert len(end_pos) == max_seq_length

        features.append(
            InputFeature(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_ids=start_pos,
                end_ids=end_pos,
                label_ids=label_map[example.entity_label],
                input_len=input_len
            ))

    return features


class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "query_train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "query_dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "query_test.json")), "test")

    def get_labels(self):
        """See base class."""
        # return ["O", "Chemical"]
        # return ["O", "BioNE"]
        # return ["O", "MATERIAL", "DEVICE", "EXPERIMENT", "VALUE"]
        # return ["O", "MAT", "SPL", "DSC", "PRO", "APL", "CMT", "SMT"]
        return ["O", "anode_material", "cathode_material", "conductivity", "current_density",
                "degradation_rate", "device", "electrolyte_material", "fuel_used", "interlayer_material",
                "open_circuit_voltage", "power_density", "resistance", "support_material",
                "time_of_operation", "thickness", "voltage", "working_temperature"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            query_item = line['query_item']
            context_item = line['context_item']
            start_position = line['start_position']
            end_position = line['end_position']
            entity_label = line['entity_label']
            # print(text_a)
            # subject = get_entities(labels, id2label=None, markup='bio')
            examples.append(
                InputExample(guid=guid, query_item=query_item, context_item=context_item, start_position=start_position,
                             end_position=end_position, entity_label=entity_label))
        return examples


ner_processors = {
    "cluener": CluenerProcessor
}
