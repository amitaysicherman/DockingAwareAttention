from utils import ECType, tokens_to_canonical_smiles
from dataset import SeqToSeqDataset
from preprocessing.tokenizer_utils import TOKENIZER_DIR, get_ec_tokens

from transformers import T5Config
from transformers import PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from model import CustomT5Model
import numpy as np
import os
from tqdm import tqdm

DEBUG = False

# print available devices
print("Available devices:")
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))


def suf_to_dim(suf):
    if suf == "":
        return 2560
    if suf == "_600m":
        return 1152
    if suf == "_gn" or suf == "_gn1":
        return 3072
    if suf == "_pb" or suf == "_pb1":
        return 1024
    if suf == "_re":
        return 256
    raise ValueError(f"Unknown suffix: {suf}")


def k_name(filename, k):
    assert filename.endswith(".txt")
    return filename.replace(".txt", f"_k{k}.txt")


def eval_dataset(model, tokenizer, dataloader, output_file, all_k=[1, 3, 5]):
    k = max(all_k)
    k_to_res = {k_: [] for k_ in all_k}
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_ids = batch['id'].detach().cpu().numpy().flatten().tolist()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device).bool()
        labels = batch['labels'].to(model.device)
        emb = batch['emb'].to(model.device).float()
        scores = batch['docking_scores'].to(model.device).float()
        emb_mask = batch['emb_mask'].to(model.device).bool()
        if (emb == 0).all():
            emb_args = {}
        else:
            emb_args = {"emb": emb, "emb_mask": emb_mask, "docking_scores": scores}

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=200, do_sample=False, num_beams=k,
                                 num_return_sequences=k, **emb_args)

        for j in range(len(batch_ids)):
            mask = (labels[j] != tokenizer.pad_token_id) & (labels[j] != -100)
            label = labels[j][mask]
            label_smiles = tokens_to_canonical_smiles(tokenizer, label)
            preds_list = [tokens_to_canonical_smiles(tokenizer, opt) for opt in outputs[j * k:(j + 1) * k]]
            id_ = batch_ids[j]
            for k_ in all_k:
                is_correct = int(label_smiles in preds_list[:k_])
                k_to_res[k_].append((id_, is_correct))
    for k_ in all_k:
        with open(k_name(output_file, k_), "w") as f:
            for id_, is_correct in k_to_res[k_]:
                f.write(f"{id_},{is_correct}\n")


class EvalGen(TrainerCallback):
    def __init__(self, model, tokenizer, valid_ds, test_ds, output_base, batch_size=64, emb_dim=2560):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        args = {"batch_size": batch_size, "num_workers": 0, "shuffle": False, "drop_last": False}
        self.valid_data_loader = DataLoader(valid_ds,
                                            collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, emb_dim=emb_dim,
                                                                                    model=model),
                                            **args)
        self.test_data_loader = DataLoader(test_ds, collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, emb_dim=emb_dim,
                                                                                            model=model),
                                           **args)
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)

    def run_eval(self, epoch):
        print(epoch)
        self.model.eval()
        with torch.no_grad():
            valid_output_file = f"{self.output_base}/valid_{epoch}.txt"
            eval_dataset(self.model, self.tokenizer, self.valid_data_loader, valid_output_file)
            test_output_file = f"{self.output_base}/test_{epoch}.txt"
            eval_dataset(self.model, self.tokenizer, self.test_data_loader, test_output_file)
        self.model.train()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.run_eval(state.epoch)


def args_to_name(ec_type, daa_type, emb_dropout, add_ec_tokens, emb_suf, concat_vec):
    name = f"ec-{ec_type}_daa-{daa_type}_emb-{emb_dropout}_ectokens-{add_ec_tokens}{emb_suf}"
    if concat_vec == 1:
        name += "_concat"
    elif concat_vec == 2:
        name += "_concat2"
    return name


def get_tokenizer_and_model(ec_type, daa_type, emb_dropout, add_ec_tokens, emb_suf, concat_vec):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    if ec_type == ECType.PAPER or add_ec_tokens:
        new_tokens = get_ec_tokens()
        tokenizer.add_tokens(new_tokens)
    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      decoder_start_token_id=tokenizer.pad_token_id)

    prot_dim = suf_to_dim(emb_suf)
    model = CustomT5Model(config, daa_type, emb_dropout=emb_dropout, prot_dim=prot_dim, concat_vec=concat_vec)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, *args, emb_dim=2560, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_zero = np.zeros((1, emb_dim))

    def __call__(self, features):
        if "emb" not in features[0]:
            return super().__call__(features)

        regular_names = ["input_ids", "labels", "id"]
        features_to_batch = [{k: f[k] for k in f if k in regular_names} for f in features]
        batch = super().__call__(features_to_batch)

        emb_list = [f["emb"][0] for f in features]
        emb_list = [np.load(f)[0] if len(f) else self.emb_zero for f in emb_list]
        emb_list = [torch.tensor(e).float() for e in emb_list]
        batch["emb"] = torch.nn.utils.rnn.pad_sequence(emb_list, batch_first=True, padding_value=0.0)
        docking_scores_list = [f["docking_scores"] for f in features]
        batch["docking_scores"] = torch.nn.utils.rnn.pad_sequence(docking_scores_list, batch_first=True,
                                                                  padding_value=0.0)
        emb_masks = [torch.ones(len(f["emb"])) for f in features]
        batch["emb_mask"] = torch.nn.utils.rnn.pad_sequence(emb_masks, batch_first=True, padding_value=0)

        return batch


def main(ec_type, daa_type, batch_size, batch_size_factor, learning_rate, max_length, emb_dropout, add_ec_tokens,
         epochs, emb_suf, concat_vec):
    tokenizer, model = get_tokenizer_and_model(ec_type, daa_type=daa_type, emb_dropout=emb_dropout,
                                               add_ec_tokens=add_ec_tokens, emb_suf=emb_suf, concat_vec=concat_vec)
    common_ds_args = {"tokenizer": tokenizer, "max_length": max_length, "emb_suf": emb_suf}
    train_dataset = SeqToSeqDataset(["ecreact", "uspto"], "train", weights=[40, 1], **common_ds_args,
                                    add_emb=[True, False])

    val_dataset = SeqToSeqDataset(["ecreact"], "valid", **common_ds_args, add_emb=[True])
    test_dataset = SeqToSeqDataset(["ecreact"], "test", **common_ds_args, add_emb=[True])
    run_name = args_to_name(ec_type, daa_type, emb_dropout, add_ec_tokens, emb_suf, concat_vec)
    print(f"Run name: {run_name}")
    # Training arguments
    output_dir = f"results/{run_name}"
    resume_from_checkpoint = False
    if os.path.exists(output_dir):
        dirs_in_output = os.listdir(output_dir)
        for dir in dirs_in_output:
            if "checkpoint" in dir:
                resume_from_checkpoint = True
                break
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        warmup_steps=0,
        logging_steps=1 / (epochs * epochs),
        save_steps=1 / (epochs * epochs),
        save_total_limit=3,
        save_strategy="steps",
        eval_strategy="no",

        auto_find_batch_size=False,
        per_device_train_batch_size=batch_size,
        report_to='tensorboard',

        run_name=run_name,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        gradient_accumulation_steps=batch_size_factor,
        save_safetensors=False,
        group_by_length=True,
        local_rank=LOCAL_RANK
    )
    emb_dim = suf_to_dim(emb_suf)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[EvalGen(model, tokenizer, val_dataset, test_dataset, output_dir, emb_dim=emb_dim)],
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer, emb_dim=emb_dim, model=model, padding=True),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ec_type", default=2, type=int)
    parser.add_argument("--daa_type", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_factor", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--add_ec_tokens", type=int, default=0)
    parser.add_argument("--emb_dropout", default=0.0, type=float)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--emb_suf", type=str, default="")
    parser.add_argument("--concat_vec", type=int, default=0)

    args = parser.parse_args()
    LOCAL_RANK = args.local_rank
    ec_type = ECType(args.ec_type)
    if ec_type == ECType.NO_EC or ec_type == ECType.PAPER:
        print("Setting daa_type to 0 (NO EC or PAPER)")
        args.daa_type = 0
    main(ec_type=ec_type, daa_type=args.daa_type, batch_size=args.batch_size, batch_size_factor=args.batch_size_factor,
         learning_rate=args.learning_rate, max_length=args.max_length, emb_dropout=args.emb_dropout,
         add_ec_tokens=args.add_ec_tokens, epochs=args.epochs, emb_suf=args.emb_suf, concat_vec=args.concat_vec)
