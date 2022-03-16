
def predict(model, prediction_dataloader):
    '''
    :param model: trained bert model
    :param prediction_dataloader: tensor dataloader for predictions
    :return: the probs: sentiment score
    '''

    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    predictions, true_labels = [], []
    # Predict
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up predictio

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
        flat_preds = [item for sublist in predictions for item in sublist]
        # The scores are normalized through a softmax.
        probs = np.argmax(flat_preds, axis=1).flatten()
    return probs
