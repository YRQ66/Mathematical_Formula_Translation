
def get_pred_and_label_str(pred, label, tokenizer):
    pred_str = tokenizer.decode_batch(pred.tolist(), skip_special_tokens=True)
    label[label == -100] = tokenizer.token_to_id("[PAD]")
    label_str = tokenizer.decode_batch(label.tolist(), skip_special_tokens=True)
    return (pred_str, label_str)