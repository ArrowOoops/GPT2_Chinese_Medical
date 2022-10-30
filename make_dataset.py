
from tqdm import tqdm
from tqdm import trange
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tokenizations import tokenization_bert
tokenizer_path = "cache/vocab_small.txt"
full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=tokenizer_path)
all_ids = []

with open('D:\\VSCode\\GPT2-Chinese\\data\\train.txt', 'r', encoding='utf8') as f:
    for line in tqdm(f):
        l = line.strip()
        token_l = full_tokenizer.convert_tokens_to_ids(full_tokenizer.tokenize(l))
        if len(token_l) >= 20 and len(token_l) <= 1024:
            pad_len = 1024 - len(token_l)
            token_l.extend([0] * pad_len)
            all_ids.append(token_l)

with open('D:\\VSCode\\GPT2-Chinese\\data\\train_tok.txt', 'w', encoding='utf8') as f:
    for ids in all_ids:
        for id in ids:
            f.write(str(id) + ' ')
        f.write('\n')