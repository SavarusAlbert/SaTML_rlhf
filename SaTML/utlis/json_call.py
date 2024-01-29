import json


def read_json(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            res.append(json.loads(line))
    return res


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')

