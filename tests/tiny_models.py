"""Build tiny, randomly initialised models plus tokenizers on disk.

They let the whole pipeline run on CPU in seconds without downloading
anything, so the tests check wiring and shapes, not accuracy.
"""
import os

from tokenizers import ByteLevelBPETokenizer, processors
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, RobertaConfig,
                          RobertaForMaskedLM, RobertaForSequenceClassification,
                          RobertaTokenizerFast)

WORDS = ("the food was great terrible good bad okay and service staff place news world "
         "sports business tech it is very not i love this never going back").split()
CORPUS = [" ".join(WORDS[i:] + WORDS[:i]) + ' . " "' for i in range(len(WORDS))] * 50
CORPUS += [" World Sports Business Tech News great terrible okay"] * 50


def build(root: str) -> dict:
    os.makedirs(root, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(CORPUS, vocab_size=400, min_frequency=1,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|endoftext|>"])
    vocab = bpe.get_vocab_size()

    # GPT-2 style tokenizer + models
    gpt_dir = os.path.join(root, "tiny-gpt2")
    gtok = GPT2TokenizerFast(tokenizer_object=bpe._tokenizer.__class__.from_str(bpe._tokenizer.to_str()),
                             bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                             unk_token="<|endoftext|>")
    gtok.save_pretrained(gpt_dir)
    GPT2LMHeadModel(GPT2Config(vocab_size=vocab, n_embd=32, n_layer=2, n_head=2,
                               n_positions=256, bos_token_id=gtok.bos_token_id,
                               eos_token_id=gtok.eos_token_id)).save_pretrained(gpt_dir)

    # RoBERTa style tokenizer + models
    rob = bpe._tokenizer.__class__.from_str(bpe._tokenizer.to_str())
    rob.post_processor = processors.RobertaProcessing(("</s>", bpe.token_to_id("</s>")),
                                                      ("<s>", bpe.token_to_id("<s>")))
    rtok = RobertaTokenizerFast(tokenizer_object=rob, bos_token="<s>", eos_token="</s>",
                                sep_token="</s>", cls_token="<s>", unk_token="<unk>",
                                pad_token="<pad>", mask_token="<mask>")
    rcfg = dict(vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=64, max_position_embeddings=300,
                pad_token_id=rtok.pad_token_id, bos_token_id=rtok.bos_token_id,
                eos_token_id=rtok.eos_token_id)
    paths = {"gpt2": gpt_dir}
    for name, model in [("mlm", RobertaForMaskedLM(RobertaConfig(**rcfg))),
                        ("cls", RobertaForSequenceClassification(RobertaConfig(num_labels=2, **rcfg)))]:
        d = os.path.join(root, f"tiny-roberta-{name}")
        rtok.save_pretrained(d)
        model.save_pretrained(d)
        paths[name] = d
    return paths


if __name__ == "__main__":
    print(build("/tmp/tiny_models"))
