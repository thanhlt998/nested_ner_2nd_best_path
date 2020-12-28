import json
import os
from transformers import PhobertTokenizer, AutoTokenizer
import numpy as np


def load_jsonl(fn):
    result = []
    with open(fn, mode='r', encoding='utf8') as f:
        for line in f:
            result.append(json.loads(line.replace('\ufeff', '').strip()))
    return result


def parse_sentence(context: str, positions: dict, tokenizer: PhobertTokenizer):
    words = context.split()
    lengths = []
    all_tokens = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        all_tokens.extend(tokens)
        lengths.append(len(tokens))

    new_indices = np.cumsum([0, *lengths, 0], dtype=np.int)
    # print(new_indices)
    new_positions = {
        label: [[new_indices[pos[0]], new_indices[pos[1]]] for pos in poses]
        for label, poses in positions.items()
    }

    # print(new_positions, positions)

    labels = sum([[(tuple(pos), label) for pos in poses] for label, poses in new_positions.items()], [])
    labels = sorted(labels, key=lambda label: label[0])

    lines = [
        ' '.join(all_tokens) + '\n',
        '|'.join([f'{pos[0]},{pos[1]} E#{label}' for pos, label in labels]) + '\n',
        '\n',
    ]
    return lines


def parse_file(fn, tokenizer: PhobertTokenizer):
    sentences = load_jsonl(fn)
    lines = []
    for sentence in sentences:
        try:
            context = sentence['context']
            positions = sentence['positions']
            lines.extend(parse_sentence(context, positions, tokenizer))
        except Exception as e:
            print(sentence)
    return lines


def parse_vlsp2018(input_dir, output_dir):
    input_fns = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
    output_fns = ['train.data', 'dev.data', 'test.data']
    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")

    for input_fn, output_fn in zip(input_fns, output_fns):
        output_lines = parse_file(os.path.join(input_dir, input_fn), tokenizer)
        with open(os.path.join(output_dir, output_fn), mode='w', encoding='utf8') as f:
            f.writelines(output_lines)


if __name__ == '__main__':
    # data = load_jsonl('raw_data/vlsp2018/mrc_preprocess/test.jsonl')
    # print(data[:3])

    # with open('test_vlsp2018.data', mode='w', encoding='utf8') as f:
    #     f.writelines(parse_file('raw_data/vlsp2018/mrc_preprocess/test.jsonl'))
    
    # test parse sent
    # print(parse_sentence(data[3]['context'], data[3]['positions'], AutoTokenizer.from_pretrained("vinai/phobert-base")))
    # print(parse_sentence(
    #     "Bảo_Anh chia_sẻ dòng trạng_thái đượm buồn giữa tin_đồn chia_tay Hồ_Quang Hiếu",
    #     {'PER': [[1, 2], [10, 12]]}, AutoTokenizer.from_pretrained("vinai/phobert-base")))

    input_dir = 'raw_data/vlsp2018/preprocessed_data'
    output_dir = 'data/vlsp2018'
    parse_vlsp2018(input_dir, output_dir)
