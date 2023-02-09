import torch

@torch.no_grad()
def estimate_loss(model, data, eval_iters):
    out = {}

    # set model in evaluation phase
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        # get losses eval_iters times
        for it in range(eval_iters):
            X, Y = data.get_batch(split)
            _, loss = model(X, Y)
            losses[it] = loss.item()
        
        # get the average loss for this split
        out[split] = losses.mean()

    # reset model to training mode
    model.train()
    return out