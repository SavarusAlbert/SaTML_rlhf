import torch
from trl import PPOTrainer, PPOConfig, create_reference_model
from tqdm import tqdm


class PPOPipeline:
    def __init__(self, model, reward_model, tokenizer, data_collator, dataset, **kwargs):
        self.reward_model = reward_model
        self.model = model
        self.model_ref = create_reference_model(model)
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam()
        self.data_collator = data_collator
        self.dataset = dataset
        
        self.ppo_config = PPOConfig(**kwargs)
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.model_ref,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            data_collator=self.data_collator,
            dataset=self.dataset
            )
    
    def train(self):
        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            if epoch >= self.ppo_config.total_ppo_epochs:
                break
        
        input_tensors = batch["input_ids"]

        response_tensors = self.ppo_trainer.generate(
            input_tensors,
            return_prompt=False,
            )
        
        batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards_outputs = self.reward_model.batch_predict(texts)
        rewards = [torch.tensor(output[0]["score"]) for output in rewards_outputs]

        stats = self.ppo_trainer.step(input_tensors, response_tensors, rewards)
        self.ppo_trainer.log_stats(stats, batch, rewards)

        if self.ppo_config.save_freq and epoch and epoch % self.ppo_config.save_freq == 0:
            self.ppo_trainer.save_pretrained(self.ppo_config.output_dir + f"step_{epoch}")
        