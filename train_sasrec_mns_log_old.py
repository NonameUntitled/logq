from argparse import ArgumentParser
import os

import torch
from utils import load_config, build_model, get_device
from dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader
from tqdm import tqdm
from eval_utils import evaluate
from torchinfo import summary

models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
args = parser.parse_args()
config = load_config(args.config)

num_items = get_num_items(config.dataset_name) 
device = get_device()
model = build_model(config)

train_dataloader = get_train_dataloader(config.dataset_name, batch_size=config.train_batch_size,
                                         max_length=config.sequence_length, train_neg_per_positive=config.negs_per_pos)
val_dataloader = get_val_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)

optimiser = torch.optim.Adam(model.parameters())
batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0

model = model.to(device)
summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

if config.path_to_cnt is not None:
    import pickle
    with open(config.path_to_cnt, 'rb') as f:
        data = pickle.load(f)
    item_interactions_num = torch.tensor(data, device=device, dtype=torch.float32)  # (num_items)
    total_interraction_sum = item_interactions_num.sum()

for epoch in range(config.max_epochs):
    model.train()   
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0
    for batch_idx in pbar:
        step += 1
        positives, negatives = [tensor.to(device) for tensor in next(batch_iter)]
        model_input = positives[:, :-1]
        mask = (model_input != num_items + 1).bool()
        last_hidden_state, _ = model(model_input)
        labels = positives[:, 1:]
        labels = labels[mask]

        output_embeddings = model.get_output_embeddings()

        positive_embeddings = output_embeddings(
            labels
        )
        positive_scores = torch.einsum(
            'ad,ad->a',
            last_hidden_state[mask],
            positive_embeddings
        )[:, None].to(torch.float64)

        num_in_batch_negatives = config.negs_per_pos
        uniform_negatives = negatives[:, 1:, :int(num_in_batch_negatives // 2)]
        unique_values, indices = torch.unique(labels, sorted=False, return_inverse=True)
        negative_indices = torch.argmax(
            (torch.arange(start=0, end=unique_values.shape[0], device=device)[:, None] == indices[None]).int(), dim=1
        )
        negative_indices = negative_indices[torch.randperm(negative_indices.shape[0], device=device)[:int(num_in_batch_negatives // 2)]]
        negative_inbatch_ids = labels[negative_indices]

        mixed_negatives_ids = torch.cat([uniform_negatives[mask], negative_inbatch_ids[None].tile([labels.shape[0], 1])], dim=-1)

        in_batch_negative_embeddings = output_embeddings(mixed_negatives_ids)
        negative_scores = torch.einsum(
            'ad,and->an',
            last_hidden_state[mask],
            in_batch_negative_embeddings
        ).to(torch.float64)

        negative_mask = labels[:, None] != mixed_negatives_ids
        negative_scores[~negative_mask] = -torch.inf

        negative_correction = torch.clamp(item_interactions_num[mixed_negatives_ids] / (total_interraction_sum), min=1e-10)
        negative_scores_corrected = negative_scores - torch.log(negative_correction)

        all_scores = torch.cat([positive_scores, negative_scores_corrected], dim=-1)
        loss = (-torch.log_softmax(all_scores, dim=1))[:, 0]
        
        loss = loss.mean()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        loss_sum += loss.item()
        pbar.set_description(f"Epoch {epoch} loss: {loss_sum / (batch_idx + 1)}")

    evaluation_result = evaluate(model, val_dataloader, config.metrics, config.recommendation_limit, 
                                config.filter_rated, device=device) 
    print(f"Epoch {epoch} evaluation result: {evaluation_result}")

    if evaluation_result[config.val_metric] > best_metric:
        best_metric = evaluation_result[config.val_metric]
        model_name = f"models/sasrec-mns-logq-old-no-fix-{config.dataset_name}-step:{step}-t:{config.gbce_t}-negs:{config.negs_per_pos}-emb:{config.embedding_dim}-dropout:{config.dropout_rate}-metric:{best_metric}.pt" 
        print(f"Saving new best model to {model_name}")
        if best_model_name is not None:
            os.remove(best_model_name)
        best_model_name = model_name
        steps_not_improved = 0
        torch.save(model.state_dict(), model_name)
    else:
        steps_not_improved += 1
        print(f"Validation metric did not improve for {steps_not_improved} steps")
        if steps_not_improved >= config.early_stopping_patience:
            print(f"Stopping training, best model was saved to {best_model_name}")
            break
