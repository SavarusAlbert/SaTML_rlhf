from augments import get_args
from model import LlamaModel, iFlytekSparkModel, ChatGLMModel
from utils import prepare_data

def main():
    args = get_args()

    if args.llama_model_path:
        model = LlamaModel(model_path=args.llama_model_path)
    elif args.iflytekspark_model_path:
        model = iFlytekSparkModel()
    elif args.chatglm_model_path:
        model = ChatGLMModel()
    else:
        raise ValueError("Please specify a model path.")
    
    if args.checkpoint_path:
        model.load_state_dict()
    
    if args.do_train:
        model.train()
    elif args.do_eval:
        model.eval()
    elif args.do_predict:
        model.predict()
    else:
        raise ValueError("Please specify an operation.")
    
    train_ds, valid_ds, test_ds = prepare_data(args)

    



if __name__ == '__main__':
    main()