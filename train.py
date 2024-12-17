from utils import ECType
from dataset import SeqToSeqDataset
from preprocessing.tokenizer_utils import TOKENIZER_DIR, get_ec_tokens

from transformers import T5Config
from transformers import PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import DataCollatorForSeq2Seq
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
from model import CustomT5Model
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import json
import os
import re
from tqdm import tqdm

DEBUG = False
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")


def tokens_to_canonical_smiles(tokenizer, tokens, remove_stereo=False):
    smiles = tokenizer.decode(tokens, skip_special_tokens=True)
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


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
    def __init__(self, model, tokenizer, valid_ds, test_ds, output_base, batch_size=64):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.valid_data_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=0,
                                            collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, model=model),
                                            shuffle=False, drop_last=False)

        self.test_data_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0,
                                           collate_fn=CustomDataCollatorForSeq2Seq(tokenizer, model=model),
                                           shuffle=False, drop_last=False)
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

    # def on_train_begin(self, args, state, control, **kwargs):
    #     self.run_eval(0)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.run_eval(state.epoch)


def args_to_name(ec_type, daa_type, emb_dropout, add_ec_tokens):
    return f"ec-{ec_type}_daa-{daa_type}_emb-{emb_dropout}_ectokens-{add_ec_tokens}"


def load_pretrained_model():
    base_dir = "results/uspto"
    cp_dirs = os.listdir(base_dir)
    cp_dirs = [f for f in cp_dirs if re.match(r"checkpoint-\d+", f)]
    cp_dirs = sorted(cp_dirs, key=lambda x: int(x.split("-")[1]))
    last_cp = f"{base_dir}/{cp_dirs[-1]}"
    trainer_state_file = f"{last_cp}/trainer_state.json"
    if not os.path.exists(trainer_state_file):
        raise ValueError(f"trainer_state.json not found in {base_dir}")
    with open(trainer_state_file) as f:
        trainer_state = json.load(f)
    return trainer_state["best_model_checkpoint"]


ec_type, daa_type = daa_type, emb_dropout = emb_dropout


def get_tokenizer_and_model(ec_type, daa_type, emb_dropout, add_ec_tokens):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    if ec_type == ECType.PAPER or add_ec_tokens:
        new_tokens = get_ec_tokens()
        tokenizer.add_tokens(new_tokens)
    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      decoder_start_token_id=tokenizer.pad_token_id)
    model = CustomT5Model(config, daa_type, emb_dropout=emb_dropout)
    return tokenizer, model


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        if "emb" not in features[0]:
            return super().__call__(features)

        regular_names = ["input_ids", "labels", "id"]
        features_to_batch = [{k: f[k] for k in f if k in regular_names} for f in features]
        batch = super().__call__(features_to_batch)

        emb_list = [f["emb"] for f in features]
        batch["emb"] = torch.nn.utils.rnn.pad_sequence(emb_list, batch_first=True, padding_value=0.0)
        docking_scores_list = [f["docking_scores"] for f in features]
        batch["docking_scores"] = torch.nn.utils.rnn.pad_sequence(docking_scores_list, batch_first=True,
                                                                  padding_value=0.0)
        emb_masks = [torch.ones(len(f["emb"])) for f in features]
        batch["emb_mask"] = torch.nn.utils.rnn.pad_sequence(emb_masks, batch_first=True, padding_value=0)

        return batch


#    main(ec_type=ec_type,daa_type=args.daa_type, batch_size=args.batch_size, batch_size_factor=args.batch_size_factor,
#         learning_rate=args.learning_rate, max_length=args.max_length, emb_dropout=args.emb_dropout)
def main(ec_type, daa_type, batch_size, batch_size_factor, learning_rate, max_length, emb_dropout, add_ec_tokens):
    tokenizer, model = get_tokenizer_and_model(ec_type, daa_type=daa_type, emb_dropout=emb_dropout,
                                               add_ec_tokens=add_ec_tokens)
    common_ds_args = {"tokenizer": tokenizer, "max_length": max_length}
    train_dataset = SeqToSeqDataset(["ecreact", "uspto"], "train", weights=[20, 1], **common_ds_args,
                                    add_emb=[True, False])
    val_dataset = SeqToSeqDataset(["ecreact"], "valid", **common_ds_args, add_emb=[True])
    test_dataset = SeqToSeqDataset(["ecreact"], "test", **common_ds_args, add_emb=[True])
    run_name = args_to_name(ec_type, daa_type, emb_dropout, add_ec_tokens)
    print(f"Run name: {run_name}")
    # Training arguments
    output_dir = f"results/{run_name}"
    num_train_epochs = 5
    resume_from_checkpoint = False
    if os.path.exists(output_dir):
        dirs_in_output = os.listdir(output_dir)
        for dir in dirs_in_output:
            if "checkpoint" in dir:
                resume_from_checkpoint = True
                break
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.05,
        logging_steps=1 / (num_train_epochs * num_train_epochs),
        save_steps=1 / num_train_epochs,
        save_total_limit=3,
        save_strategy="steps",
        eval_strategy="no",

        auto_find_batch_size=False,
        per_device_train_batch_size=batch_size,
        report_to='tensorboard',

        run_name=run_name,
        learning_rate=learning_rate,
        gradient_accumulation_steps=batch_size_factor,
        save_safetensors=False,
        group_by_length=True,

    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[EvalGen(model, tokenizer, val_dataset, test_dataset, output_dir)],
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ec_type", default=3, type=int)
    parser.add_argument("--daa_type", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_factor", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--add_ec_tokens", type=int, default=0)
    parser.add_argument("--emb_dropout", default=0.0, type=float)

    args = parser.parse_args()
    ec_type = ECType(args.ec_type)
    if ec_type == ECType.NO_EC or ec_type == ECType.PAPER:
        print("Setting daa_type to 0 (NO EC or PAPER)")
        args.daa_type = 0
    main(ec_type=ec_type, daa_type=args.daa_type, batch_size=args.batch_size, batch_size_factor=args.batch_size_factor,
         learning_rate=args.learning_rate, max_length=args.max_length, emb_dropout=args.emb_dropout,
         add_ec_tokens=args.add_ec_tokens)
