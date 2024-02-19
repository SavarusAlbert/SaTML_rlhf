from augments import get_args
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from utils import prepare_data
from trainer import PPOPipeline


def main():
    args = get_args()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.llama_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model_path)
    reward_model = None

    train_ds, test_ds = prepare_data(args)
    data_collator = lambda x: x

    trainer = PPOPipeline(model, reward_model, tokenizer, data_collator, train_ds)

    trainer.train()


if __name__ == '__main__':
    main()