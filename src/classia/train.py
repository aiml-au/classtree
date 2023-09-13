import logging
import json
import os
import collections
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.nn.functional import softmax
from torchvision.transforms.v2 import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomGrayscale, ToImageTensor, ConvertImageDtype, Normalize
# from torchtext.models.roberta.bundler import ROBERTA_BASE_ENCODER
from .models import get_text_encoder
import matplotlib.pyplot as plt
from tqdm import tqdm

from .hier import truncate_given_lca, FindLCA, SumDescendants
from .dataset import ClassiaImageDataset, ClassiaTextDataset
from .loss import MarginLoss
from .metrics import UniformLeafInfoMetric, DepthMetric, IsCorrect
from .predict import pareto_optimal_predictions, argmax_with_confidence

LOGGER = logging.getLogger(__name__)

PATIENCE = 5 # 10
SPLIT_RATIO = 0.8
MIN_THRESHOLD = 0.5

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError


def write_data(tree, label_set, files, labels, meta_data, models_dir, model_name):

    with open(f'{models_dir}/{model_name}/tree.csv', 'w') as f:
        f.write('\n'.join(f'{parent},{child}' for parent, child in tree.edges()))
    with open(f'{models_dir}/{model_name}/labels.txt', 'w') as f:
        f.write('\n'.join(label_set))
    with open(f'{models_dir}/{model_name}/train.csv', 'w') as f:
        f.write('\n'.join(f'{file},{label}' for file, label in zip(files, labels)))

    with open(f'{models_dir}/{model_name}/meta.json', 'w') as f:
        json.dump(meta_data, f, indent=4)


def get_dataset(model_type, files, labels, model_size='base'):
    if model_type =='image':
        transform = Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomGrayscale(),
            ToImageTensor(),
            ConvertImageDtype(torch.float),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.244, 0.225]
            )
        ])
        dataset = ClassiaImageDataset(files, labels, transform=transform)

    elif model_type =='text':
        text_encoder= get_text_encoder(model_size)
        transform = text_encoder.transform()
        dataset = ClassiaTextDataset(files, labels, transform=transform)

    else:
        raise ValueError(f'{model_type} type of model is not implemented.')
    
    return dataset
    
   
def get_dataloader(model_type, dataset, batch_size=8, model_size='base'):

    def pad_tokens(batch):
        max_tokens = max((seq.shape[0] for seq, _, _ in batch))

        input_ids = torch.full((len(batch), max_tokens), fill_value=get_text_encoder(model_size).encoderConf.padding_idx)
        for i in range(len(batch)):
            input_ids[i, :len(batch[i][0])] = batch[i][0]

        labels = torch.tensor([label for _, _, label in batch], dtype=torch.long)

        return input_ids, labels
        
    collate_fn = {'collate_fn': pad_tokens} if model_type =='text' else {}

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **collate_fn)

    return data_loader


def prepare_dataloaders(model_type, dataset, batch_size=8, model_size='base'):
    train_dataset, eval_dataset = random_split(dataset,(SPLIT_RATIO, 1-SPLIT_RATIO))

    train_loader = get_dataloader(model_type, train_dataset, batch_size, model_size)
    eval_loader = get_dataloader(model_type, eval_dataset, batch_size, model_size)
    return train_loader, eval_loader


# Save training/validationloss plots
def save_training_plot(train_loss, val_loss, save_dir):
    print('Saving loss plots...')

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{save_dir}/loss_plot.png")
    plt.show()


LabelMap = collections.namedtuple('LabelMap', ['to_node', 'to_target'])

def get_metric_fns(tree):
    info_metric = UniformLeafInfoMetric(tree)
    depth_metric = DepthMetric(tree)
    metric_fns = {
        'exact': lambda gt, pr: pr == gt,
        'correct': IsCorrect(tree),

        'info_recall': info_metric.recall,
        'info_precision': info_metric.precision,

        'depth_recall': depth_metric.recall,
        'depth_precision': depth_metric.precision,
    }
    return metric_fns    


def get_label_map(tree):
    label_map = LabelMap(
                to_node=np.arange(tree.num_nodes()),
                to_target=np.arange(tree.num_nodes())) # to_target=tree.leaf_subset() when training only on leaf node

    # Convert target map to torch tensor.
    label_map = LabelMap(
        to_node=label_map.to_node,
        to_target=torch.from_numpy(label_map.to_target))
    
    return label_map

def train_model(model, train_loader, eval_loader, tree, label_set, models_dir, model_name, model_size, epochs=50, lr=0.001, resume=False, device='cuda'):

    specificity = -tree.num_leaf_descendants()
    not_trivial = (tree.num_children() != 1)

    train_label_map = eval_label_map = get_label_map(tree)

    model.to(device)
    loss_fn = MarginLoss(tree, with_leaf_targets=False, device=device)

    optimizer = SGD(model.parameters(), lr=lr)

    best_loss = float('inf') 
    best_epoch = 0
    best_metric = None
    train_loss_history, eval_loss_history = [], []

    items_to_log = dict.fromkeys(['epoch', 'eval_loss', 'train_loss'])
    training_logs = {'best': items_to_log, 'logs': items_to_log} 

    def add_meta_to_state(state, epoch):
        model_dict = {'epoch': epoch,
                    'labels':label_set,
                    'tree': tree, 
                    'model_size': model_size,                     
                    'model_weight_name': model._get_name(),                     
                    'model_state_dict': state
        }
        return model_dict
    
    metric_fns = get_metric_fns(tree)

    # Loading pretrained weights and resume from last epoch | condition: resume
    start_epoch = 0
    if resume:
        if os.path.exists(f'{models_dir}/{model_name}/best.pth'):
            checkpoint = torch.load(f'{models_dir}/{model_name}/best.pth')

            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            LOGGER.info(f'Resuming training from last best epoch {start_epoch}.')
        else:
            LOGGER.warning('Saved best model does not exist in this directory.')
    
    end_epoch = start_epoch+epochs

    for epoch in range(start_epoch+1, end_epoch+1):
        total_train_loss =  0.
        model.train()        

        for inputs, gt_labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{end_epoch}", position=1):
            inputs = inputs.to(device)

            gt_targets = train_label_map.to_target[gt_labels]
            assert torch.all(gt_targets >= 0)
            optimizer.zero_grad()
            theta = model(inputs)

            loss = loss_fn(theta, gt_targets.to(device)) 
            
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            total_train_loss += train_loss

        mean_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(mean_train_loss)

        ######################    
        # validate the model #
        ######################
        find_lca = FindLCA(tree)
        metric_totals = {field: 0 for field in metric_fns}  
        with torch.no_grad():
            model.eval() 
            total_eval_loss = 0.

            for inputs, gt_labels in tqdm(eval_loader, desc=f"Eval Epoch {epoch}/{end_epoch}", position=1):
                inputs = inputs.to(device)
                theta = model(inputs) 

                if eval_label_map.to_target is not None:
                    gt_targets = eval_label_map.to_target[gt_labels]
                    assert torch.all(gt_targets >= 0)

                loss = loss_fn(theta, gt_targets.to(device))
                eval_loss = loss.item()
                total_eval_loss += eval_loss

                # metric
                def pred_fn(theta):
                    return SumDescendants(tree, strict=False).to(device)(softmax(theta, dim=-1), dim=-1)

                prob = pred_fn(theta).cpu().numpy()
                gt_node = eval_label_map.to_node[gt_labels]

                pr = argmax_with_confidence(specificity, prob, MIN_THRESHOLD, not_trivial) # TODO: check and compare with pareto               
                pred = truncate_given_lca(gt_node, pr, find_lca(gt_node, pr))
                
                for field in metric_fns:
                    metric_fn = metric_fns[field]
                    metric_value = metric_fn(gt_node, pred)
                    metric_totals[field] += metric_value.sum()                    

        mean_eval_loss = total_eval_loss / len(eval_loader)
        eval_loss_history.append(mean_eval_loss)        

        LOGGER.info('Train Loss: {} | Eval Loss: {} at Epoch:{}'.format(mean_train_loss, epoch, mean_eval_loss, epoch)) 

        # Bundle hierarchy and metadata
        curent_state = add_meta_to_state(model.state_dict(), epoch) 
        # Save latest model
        torch.save(curent_state, f'{models_dir}/{model_name}/latest.pth')

        # Show and save metric only at best validation epoch
        def show_metric():
            metric_means = {field: metric_totals[field] / len(eval_loader)
                                        for field in metric_totals}         
            LOGGER.info('[Evaluation Metric]:')
            for metric, values in metric_means.items(): 
                LOGGER.info(f'\t{metric}: {"{:.2f}%".format(values)}')

            return metric_means

        # Save best model at best evaluation loss 
        if mean_eval_loss < best_loss:
            best_loss = mean_eval_loss
            best_epoch = epoch
            metric_means = show_metric() 
            best_metric = metric_means
            training_logs['best']['epoch'] = best_epoch 
            training_logs['best']['eval_loss'] = mean_eval_loss
            training_logs['best']['train_loss'] = mean_train_loss
            training_logs['best']['best_metric'] = best_metric

            print(f'Saving best model at Epoch {best_epoch} | val_loss: {best_loss}')
            torch.save(curent_state, f'{models_dir}/{model_name}/best.pth')
        
        training_logs['logs']['epoch'] = epoch 
        training_logs['logs']['eval_loss'] = mean_eval_loss
        training_logs['logs']['train_loss'] = mean_train_loss

        # Early stopping        
        if best_epoch <= epoch - PATIENCE: # nothing is improving for a while 
            print(f"Early stopping.. at Epoch {epoch} with patience of {PATIENCE} epochs") 
            break 
        
    LOGGER.info('TRAINING COMPLETE')
    save_training_plot(train_loss_history, eval_loss_history, f'{models_dir}/{model_name}')


    with open(f'{models_dir}/{model_name}/logs.json', 'w') as f:
        json.dump(training_logs, f, indent=4)


def evaluate(model, eval_loader, tree, models_dir, model_name, device='cuda'):

    if os.path.exists(f'{models_dir}/{model_name}/best.pth'):
        checkpoint = torch.load(f'{models_dir}/{model_name}/best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        LOGGER.warning('Saved best model does not exist in this directory.')
        return
    
    model.to(device)

    specificity = -tree.num_leaf_descendants()
    not_trivial = (tree.num_children() != 1)

    eval_label_map = get_label_map(tree)
    metric_fns = get_metric_fns(tree)

    find_lca = FindLCA(tree)
    metric_totals = {field: 0 for field in metric_fns}  
    with torch.no_grad():
        model.eval() 

        for inputs, gt_labels in tqdm(eval_loader, position=1):
            inputs = inputs.to(device)
            theta = model(inputs)

            # metric
            def pred_fn(theta):
                return SumDescendants(tree, strict=False).to(device)(softmax(theta, dim=-1), dim=-1)

            prob = pred_fn(theta).cpu().numpy()
            gt_node = eval_label_map.to_node[gt_labels]

            pr = argmax_with_confidence(specificity, prob, MIN_THRESHOLD, not_trivial) # TODO: check and compare with pareto               
            pred = truncate_given_lca(gt_node, pr, find_lca(gt_node, pr))
            
            for field in metric_fns:
                metric_fn = metric_fns[field]
                metric_value = metric_fn(gt_node, pred)
                metric_totals[field] += metric_value.sum()                    

    # Show and save metric only at best validation epoch
    def show_metric():
        metric_means = {field: metric_totals[field] / len(eval_loader)
                                    for field in metric_totals}         
        LOGGER.info('[Evaluation Metric]:')
        for metric, values in metric_means.items(): 
            LOGGER.info(f'\t{metric}: {"{:.2f}%".format(values)}')

        return metric_means
    
    _ = show_metric()
