from datasets import load_metric
cer_metric = load_metric("cer")

def compute_cer(pred_ids, label_ids, tokenizer):
    pred_str = tokenizer.decode_batch(pred_ids.tolist(), skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.token_to_id("[PAD]")
    label_str = tokenizer.decode_batch(label_ids.tolist(), skip_special_tokens=True)
    
    # Filter out empty label strings
#    for l in enumerate(label_str):
#        if not l:
#            label_str.pop(l)
#            pred_str.pop(l)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

def get_pred_and_label_str(pred, label, tokenizer):
    pred_str = tokenizer.decode_batch(pred.tolist(), skip_special_tokens=True)
    label[label == -100] = tokenizer.token_to_id("[PAD]")
    label_str = tokenizer.decode_batch(label.tolist(), skip_special_tokens=True)
    return (pred_str, label_str)