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


def prepare_data(args):
    train_ds = read_json(args.train_data_path)
    test_ds = read_json(args.eval_data_path)
    
    return train_ds, test_ds