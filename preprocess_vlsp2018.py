from lxml import etree
from vncorenlp import VnCoreNLP
import re
from collections import defaultdict
import os
import json

annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)

s = """Chiều 22-9 tại <ENAMEX TYPE="LOCATION">TPHCM</ENAMEX>, <ENAMEX TYPE="ORGANIZATION">Trung tâm Hỗ trợ phụ nữ và Chăm sóc sức khỏe sinh sản</ENAMEX> - <ENAMEX TYPE="ORGANIZATION">Hội Liên hiệp Phụ nữ <ENAMEX TYPE="LOCATION">Việt Nam</ENAMEX></ENAMEX> đã tổ chức buổi hội thảo “Tác hại nghiêm trọng của chất coumarin trong thuốc lá nhập lậu đối với sức khỏe thai phụ và thai nhi”."""
s2 = """Như <ENAMEX TYPE="ORGANIZATION">Thanh Niên</ENAMEX> đã đưa tin, sáng 17.9, bệnh nhân <ENAMEX TYPE="PERSON">Đ.</ENAMEX> đến <ENAMEX TYPE="ORGANIZATION">Bệnh viện (BV) phẫu thuật thẩm mỹ EMCSA</ENAMEX> (<ENAMEX TYPE="LOCATION">số 14/27 Hoàng Dư Khương</ENAMEX>, <ENAMEX TYPE="LOCATION">P.12</ENAMEX>, <ENAMEX TYPE="LOCATION">Q.10</ENAMEX>, <ENAMEX TYPE="LOCATION">TP.HCM</ENAMEX>) mổ chỉnh xương hai hàm hai bên. Ca mổ do bác sĩ <ENAMEX TYPE="PERSON">T.N.Q.P</ENAMEX> thực hiện."""

tree = etree.XML("<root>" + s + "</root>")

root: etree._Element = tree.xpath('/root')[0]


def parse(node: etree._Element, n_prev_tokens=0, ):
    n_tokens = 0
    for e in node.xpath("./text()|*"):
        if type(e) is etree._ElementUnicodeResult:
            tokens = str(e).strip().split()
            n_tokens += len(tokens)
        else:
            n_tokens += parse(e, n_prev_tokens=n_prev_tokens + n_tokens)

    node.set('start_pos', str(n_prev_tokens))
    node.set('end_pos', str(n_prev_tokens + n_tokens))

    return n_tokens


def encode_tag(doc):
    doc = re.sub(r'<ENAMEX TYPE="(\w+)">', r' entity\1 ', doc)
    return re.sub(r'</ENAMEX>', ' entityclosetag ', doc)


def decode_tag(doc):
    doc = re.sub(r'entityclosetag', '</ENAMEX>', doc)
    doc = re.sub(r'entity(\w+)', r'<ENAMEX TYPE="\1">', doc)
    return doc


def remove_tag(doc):
    doc = re.sub(r'<[^>]+>', ' ', doc).strip()
    return re.sub(r'\s+', ' ', doc)


def parse_sentence(sent):
    sent = re.sub('&', '&amp;', sent)
    xml_tree = etree.XML(f"<root>{sent}</root>")
    root: etree._Element = xml_tree.xpath('/root')[0]
    parse(root)
    labels = defaultdict(list)
    for e in root.iter('ENAMEX'):
        e: etree._Element
        start_pos = int(e.get('start_pos'))
        end_pos = int(e.get('end_pos'))
        labels[e.get('TYPE')[:3]].append([start_pos, end_pos])
    return dict(labels)


def merge_sentences(sentences):
    fix_sentences = []
    curr_sentence = sentences[0]
    for sentence in sentences[1:]:
        try:
            etree.XML(re.sub('&', '&amp;', curr_sentence))
            fix_sentences.append(curr_sentence)
            curr_sentence = sentence
        except Exception as e:
            curr_sentence = ' '.join([curr_sentence, sentence])
    fix_sentences.append(curr_sentence)
    return fix_sentences


def parse_document(doc):
    doc = re.sub('[\ufeff\xa0]', '', doc).strip()
    doc = encode_tag(doc)
    sentences = annotator.tokenize(doc)
    sentences = [
        decode_tag(' '.join(sent))
        for sent in sentences
    ]
    sentences = merge_sentences(sentences)

    result = []
    for sent in sentences:
        try:
            result.append((remove_tag(sent), parse_sentence(sent)))
        except Exception as e:
            raise Exception(f'error in {sent}')

    return sentences, result


def read(fn):
    with open(fn, mode='r', encoding='utf8') as f:
        text = f.read().strip()
    return re.split(r'\n+', text)


def preprocess(folder, output_fn, corpus_fn=None):
    file_names = []
    for root, sub_dirs, files in os.walk(folder):
        if len(files) == 0: continue
        for file in files:
            file_names.append(os.path.join(root, file))

    result = []
    corpus = []
    for fn in file_names:
        lines = read(fn)
        try:
            for line in lines:
                cor, res = parse_document(line)
                result.extend(res)
                corpus.extend(cor)
        except Exception as e:
            print(e)
            print(fn)
            print(lines)
            print('-' * 50)

    with open(output_fn, mode='w', encoding='utf8') as f:
        for context, positions in result:
            f.write(json.dumps({
                'context': context,
                'positions': positions,
            }, ensure_ascii=False))
            f.write('\n')

    if corpus_fn:
        with open(corpus_fn, mode='w', encoding='utf8') as f:
            for sent in corpus:
                f.write(sent)
                f.write('\n')


# parse(root)
# print(parse(root))
# print(etree.tostring(root, pretty_print=True, encoding='utf8').decode('utf8'))

# result_parse = parse_document(doc=s2)
# for sent, labels in result_parse:
#     print(sent)
#     print(labels)
#     print('-' * 50)

# for sent in read('raw_data/vlsp2018/raw/VLSP2018-NER-train-Jan14/Doi song/23351436.muc'):
#     print(sent)

if __name__ == '__main__':
    raw_dir = 'raw_data/vlsp2018/raw'
    input_dirs = ['VLSP2018-NER-train-Jan14', 'VLSP2018-NER-dev', 'VLSP2018-NER-Test-Domains']
    output_dir = 'raw_data/vlsp2018/preprocessed_data'
    output_fns = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
    corpus_dir = 'raw_data/vlsp2018/preprocessed_corpus'
    corpus_fns = ['train.txt', 'dev.txt', 'test.txt']

    for input_dir, output_fn, corpus_fn in zip(input_dirs, output_fns, corpus_fns):
        preprocess(os.path.join(raw_dir, input_dir), output_fn=os.path.join(output_dir, output_fn),
                   corpus_fn=os.path.join(corpus_dir, corpus_fn))
