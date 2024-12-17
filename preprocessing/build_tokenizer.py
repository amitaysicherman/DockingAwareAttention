from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
import argparse
from preprocessing.tokenizer_utils import PAD, EOS, UNK, SPACIAL_TOKENS, TOKENIZER_DIR


def create_word_tokenizer(file_paths):
    all_word = set()
    words_dict = {**SPACIAL_TOKENS}
    no_ec_words = set()
    ec_words = set()

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
        for line in lines:
            all_word.update(line.split())
    for word in all_word:
        if word.startswith("[") and word[1] in ["v", "u", "t", "q"]:
            ec_words.add(word)
        else:
            no_ec_words.add(word)
    for word in no_ec_words:
        words_dict[word] = len(words_dict)
    vocab, ec_tokens = words_dict, list(ec_words)

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_process = None
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD,
    )
    return fast_tokenizer, ec_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    file_paths = []
    datasets = ['ecreact', 'uspto']
    for dataset in datasets:
        for split in ['train', 'valid', 'test']:
            for side in ['src', 'tgt']:
                file_paths.append(f"datasets/{dataset}/{side}-{split}.txt")

    word_tokenizer, ec_tokens = create_word_tokenizer(file_paths)
    print(word_tokenizer)

    test_text = "O = S ( = O ) ( [O-] ) S"
    print(f"OR: {test_text}")
    encoded = word_tokenizer.encode(test_text, add_special_tokens=False)
    print(f"EN: {encoded}")
    decoded = word_tokenizer.decode(encoded, clean_up_tokenization_spaces=False, skip_special_tokens=True)
    print(f"DE: {decoded}")
    assert decoded == test_text
    word_tokenizer.save_pretrained(TOKENIZER_DIR)
    with open(f"{TOKENIZER_DIR}/ec_tokens.txt", "w") as f:
        f.write("\n".join(ec_tokens))
