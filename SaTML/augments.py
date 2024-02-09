import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_model_path', type=str, default='')
    parser.add_argument('--iflytekspark_model_path', type=str, default='')
    parser.add_argument('--chatglm_model_path', type=str, default='')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--train_data_path', type=str, default='')
    parser.add_argument('--eval_data_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--checkpoint_path', type=str, default='')

    args = parser.parse_args()

    return args