def bert_predict(model_name, dataloader):
    '''
    :param model_name:
    :param dataloader:
    :return:
    '''
    # load model
    checkpoint = torch.load(output_model, map_location='cpu')
    bert_classifier.load_state_dict(checkpoint['bert_classifier_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    bert_classifier.eval()
    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        # Compute logits
        with torch.no_grad():
            logits = bert_classifier(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs