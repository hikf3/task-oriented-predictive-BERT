import torch
import torch.nn as nn
#from optimizers import NoamOptimizer
from modeling import BertConfig, BertForPreTraining
from dataloader import InputFeatures, BERTdataEHR, BERTdataEHRloader
from log_utils import make_run_name, make_logger, make_checkpoint_dir
import pickle as pkl
import time
import logging
import argparse
from tqdm import tqdm, trange
import sys
import pandas as pd
import random
import numpy as np
import os
from os.path import join
from datetime import datetime
import wandb
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score

experiment='pretrain-input_visit_time'
#parameter_3753168

wandb.login(key='#addkeyhere')

run=wandb.init(
    project="ehr_bert-"+experiment,
    config={
        "model":"BERT-PRETRAIN",
        "Optimizer": "AdamW",
        "loss_fn":"CrossEntropy",
        "epochs": "500",
        "pretrain datasize":"43686",
        "hidden_size":"128", 
        "Experiment":experiment,
    }
)



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Check GPU status
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logging.info('CUDA STATUS: %s', use_cuda)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s).")
    #logging.info(f"Found {device_count} CUDA device(s).")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
        #logging.info(f"Device {i}: {device_name}")
else:
    print("CUDA is not available.")
    #logging.info("CUDA is not available.")


SAVE_FORMAT = 'epoch={epoch:0>3}-train_loss={train_loss:<.3}-train_metrics={train_metrics}-val_metrics={val_metrics}.pth'

LOG_FORMAT = (
    "Epoch: {epoch:>3} "
    "Progress: {progress:<.1%} "
    "Elapsed: {elapsed} "
    "Examples/second: {per_second:<.1} "
    "Train Loss: {train_loss:<.6} "
    "MLM Loss: {mlm_loss:<.6} "
    "NSP Loss: {nsp_loss:<.6} "
    "Train Metrics: {train_metrics} "
    "Valid Loss: {val_loss:<.6} "
    "Valid Metrics: {val_metrics} "
    "Learning rate: {current_lr:<.4} "
)
RUN_NAME_FORMAT = (
    "BERT-"
    "{phase}-"
    "num_hidden_layers={num_hidden_layers}-"
    "hidden_size={hidden_size}-"
    "num_attention_heads={num_attention_heads}-"
    "{timestamp}"
)

class BertPretrainingCriterion(torch.nn.Module):

    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()

        # Loss functions
        self.loss_fn_mask = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_fn_next = torch.nn.CrossEntropyLoss()
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_label):
        masked_lm_loss = self.loss_fn_mask(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        
        next_sentence_loss = self.loss_fn_next(seq_relationship_score, next_sentence_label)
        
        total_loss = masked_lm_loss + next_sentence_loss
        
        return masked_lm_loss, next_sentence_loss, total_loss

def calculate_accuracy(prediction_scores, seq_relationship_score, input_ids, masked_lm_labels, next_sentence_label):
    with torch.no_grad():
        # MLM accuracy
        mlm_predictions = prediction_scores.argmax(dim=-1)
        masked_positions = (masked_lm_labels == -1)
        mlm_targets = input_ids[masked_positions]
        mlm_correct = torch.eq(mlm_predictions[masked_positions], mlm_targets).sum().item()
        mlm_total = masked_positions.sum().item()
        mlm_accuracy = mlm_correct / mlm_total

        # NSP accuracy
        nsp_predictions = seq_relationship_score.argmax(dim=-1)
        nsp_correct = torch.eq(nsp_predictions, next_sentence_label).sum().item()
        nsp_total = nsp_predictions.numel()
        nsp_accuracy = nsp_correct / nsp_total

        # NSP AUC
        nsp_probs = torch.softmax(seq_relationship_score, dim=-1)[:, 1].cpu().numpy()
        nsp_auc = roc_auc_score(next_sentence_label.cpu().numpy(), nsp_probs)

    return mlm_accuracy, nsp_accuracy, nsp_auc


class BertTrainer:
    def __init__(self, model, loss_model, train_dataloader, val_dataloader,
                 with_cuda, optimizer, scheduler, clip_grads,
                 logger, checkpoint_dir, print_every, save_every):
        
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        self.model = model.to(self.device)
        self.loss_model = loss_model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grads = clip_grads

        self.logger = logger
        self.checkpoint_dir = checkpoint_dir

        self.print_every = print_every
        self.save_every = save_every

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_output_path = None

    def run_epoch(self, train_dataloader, mode='train'):

        epoch_loss = 0
        epoch_mlm_loss = 0
        epoch_nsp_loss = 0
        epoch_count = 0
        epoch_metrics = [0, 0]
        
        train_iter = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=False,
            total=len(train_dataloader))

        for batch_count, batch in enumerate(train_iter):
                #0. batch_data will be sent into the device(GPU or cpu)
                batch = {key: value.to(self.device) for key, value in batch.items()}

                #input_shape = batch['input_ids'].shape
                #print('#####input ids shape',input_shape)
                #sys.exit()
    
                #1. forward the next_sentence_prediction and masked_lm model
                prediction_scores, seq_relationship_score = self.model(input_ids=batch['input_ids'], 
                                                                visit_ids = batch['visit_segment_ids'],
                                                                timetonext_ids =batch['timetonext_ids'], 
                                                                attention_mask=batch['input_mask'], 
                                                                masked_lm_labels=batch['masked_input_ids'], 
                                                                next_sentence_label=batch['label_id'])
                # extracting labels from data
                masked_lm_labels = batch['masked_input_ids'] 
                next_sentence_label = batch['label_id']
                input_ids=batch['input_ids']
            
                #2. calculate mlm loss, nsp loss, total_loss
                masked_lm_loss, next_sentence_loss, total_loss = self.loss_model(prediction_scores, 
                                                                            seq_relationship_score,
                                                                            masked_lm_labels=batch['masked_input_ids'], 
                                                                            next_sentence_label=batch['label_id'] )
                                       
                
                mlm_losses = (masked_lm_loss).to(self.device)
                nsp_losses = (next_sentence_loss).to(self.device)
                batch_losses = total_loss.to(self.device)


                #print(f'Batch Losses: {batch_losses}')
                #print(f'MLM Losses: {mlm_losses}')
                #print(f'NSP Losses: {nsp_losses}')

                mlm_loss = mlm_losses.mean()
                nsp_loss = nsp_losses.mean()
                batch_loss = batch_losses.mean()
                
                #print(f'Batch Loss: {batch_loss}')
                #print(f'MLM Loss: {mlm_loss}')
                #print(f'NSP Loss: {nsp_loss}')

                if mode == 'train':
                    
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                    self.scheduler.step()
                    torch.cuda.synchronize()

                if epoch_count + batch_count != 0:
                    epoch_loss = (epoch_loss * epoch_count + batch_loss.item() * batch_count) / (epoch_count + batch_count)
                    epoch_mlm_loss = (epoch_mlm_loss * epoch_count + mlm_loss.item() * batch_count) / (epoch_count + batch_count)
                    epoch_nsp_loss = (epoch_nsp_loss * epoch_count + nsp_loss.item() * batch_count) / (epoch_count + batch_count)

                    #print(f'Epoch Batch Loss: {batch_loss}')
                    #print(f'Epoch MLM Loss: {mlm_loss}')
                    #print(f'Epoch NSP Loss: {nsp_loss}')

            
                    mlm_acc, nsp_acc, nsp_auc = calculate_accuracy(prediction_scores, seq_relationship_score,  input_ids, masked_lm_labels, next_sentence_label)
                    batch_metrics = [mlm_acc, nsp_acc, nsp_auc]
                    #print(f'Batch Metrics: {batch_metrics}')
                    
                    epoch_metrics = [(epoch_metric * epoch_count + batch_metric * batch_count) / (epoch_count + batch_count)
                                    for epoch_metric, batch_metric in zip(epoch_metrics, batch_metrics)]
                    #print(f'Epoch Metrics: {epoch_metrics}')
                else:
                     epoch_loss += batch_loss.item()
                     epoch_mlm_loss += mlm_loss.item()
                     epoch_nsp_loss += nsp_loss.item()
                     
                     #print(f'Epoch Batch Loss: {batch_loss}')
                     #print(f'Epoch MLM Loss: {mlm_loss}')
                     #print(f'Epoch NSP Loss: {nsp_loss}')
                    
                     mlm_acc, nsp_acc, nsp_auc = calculate_accuracy(prediction_scores, seq_relationship_score,  input_ids, masked_lm_labels, next_sentence_label)
                     batch_metrics = [mlm_acc, nsp_acc, nsp_auc]
                     epoch_metrics = batch_metrics
                    
                     #print(f'Batch Metrics: {batch_metrics}')
                     #print(f'Epoch Metrics: {epoch_metrics}')

                epoch_count += batch_count       

        return epoch_loss, epoch_mlm_loss, epoch_nsp_loss, epoch_metrics
        

    def run(self, epochs=500):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            self.model.train()

            epoch_start_time = datetime.now()     
            #train_epoch_loss,  train_epoch_metrics = self.run_epoch(self.train_dataloader, mode='train')
            train_epoch_loss, mlm_loss, nsp_loss,  train_epoch_metrics = self.run_epoch(self.train_dataloader, mode='train')
            epoch_end_time = datetime.now()

            self.model.eval()
            with torch.no_grad(): 
               val_epoch_loss, val_mlm_loss, val_nsp_loss,  val_epoch_metrics = self.run_epoch(self.val_dataloader, mode='val') 
    
            if epoch % self.print_every == 0 and self.logger:
                per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                #current_lr = self.scheduler.get_last_lr()[0]
                log_message = LOG_FORMAT.format(epoch=epoch,
                                                progress=epoch / epochs,
                                                per_second=per_second,
                                                train_loss=train_epoch_loss,
                                                mlm_loss = mlm_loss,
                                                nsp_loss = nsp_loss,
                                                val_loss=val_epoch_loss,
                                                train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
                                                val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
                                                current_lr=current_lr,
                                                elapsed=self._elapsed_time()
                                                )

                self.logger.info(log_message)
                wandb.log({"epoch": epoch,
                           "progress": epoch / epochs,
                           "per_second": per_second,
                           "train total loss": train_epoch_loss,
                           "train mlm loss": mlm_loss,
                           "train nsp loss": nsp_loss,
                           "val total loss": val_epoch_loss,
                           "val mlm loss": val_mlm_loss,
                           "val nsp loss": val_nsp_loss,
                           "train mlm accuracy": round(train_epoch_metrics[0],4),
                           "train nsp accuracy": round(train_epoch_metrics[1],4),
                           "train nsp auc": round(train_epoch_metrics[2],4),
                           "val mlm accuracy": round(val_epoch_metrics[0],4),
                           "val nsp accuracy": round(val_epoch_metrics[1],4),
                           "val nsp auc": round(val_epoch_metrics[2],4),
                           "current_lr": current_lr,
                           "elapsed": self._elapsed_time()})
                

            if epoch % self.save_every == 0:
                self._save_model(epoch, train_epoch_loss, mlm_loss, nsp_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics)

    def _save_model(self, epoch, train_epoch_loss, mlm_loss, nsp_loss,val_epoch_loss, train_epoch_metrics, val_epoch_metrics):
        
        elapsed=self._elapsed_time()
        checkpoint_name = SAVE_FORMAT.format(
            epoch=epoch,
            train_loss= train_epoch_loss,
            train_metrics='-'.join(['{:<.3}'.format(v) for v in train_epoch_metrics]),
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        checkpoint_output_path = join(self.checkpoint_dir, checkpoint_name)

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'train_metrics': train_epoch_metrics,
            'mlm_loss': mlm_loss,
            'nsp_loss': nsp_loss,
            'val_loss': val_epoch_loss,
            'val_metrics': val_epoch_metrics,
            'checkpoint': checkpoint_output_path,
            'time cost': elapsed
        }
        
        if epoch >= 0:
            self.history.append(save_state)
            with open(self.checkpoint_dir+'output_history.txt', 'a+') as f:
                f.write(str(save_state) + '\n')

        if hasattr(self.model, 'module'):  # DataParallel
            save_state['state_dict'] = self.model.module.state_dict()
        else:
            save_state['state_dict'] = self.model.state_dict()

        torch.save(save_state, checkpoint_output_path)

        representative_val_metric = val_epoch_metrics[0]
        if self.best_val_metric is None or self.best_val_metric < representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_metrics_at_best = val_epoch_metrics
            self.val_loss_at_best = val_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.train_loss_at_best = train_epoch_loss
            self.best_checkpoint_output_path = checkpoint_output_path
            self.best_epoch = epoch

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_output_path))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_output_path))

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds

def pretrain():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-vl", "--val_dataset", required=True, type=str, help="valid dataset for train bert")
    parser.add_argument("-config", "--config_path", required=True, type=str, default=str, help="config json file")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
   
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--log_output', type=str, default=None)
    
    parser.add_argument("-hs", "--hidden_size", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--num_hidden_layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--num_attention_heads", type=int, default=6, help="number of attention heads")
    parser.add_argument("-s", "--max_len", type=int, default=512, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")
    
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false") 
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument('--clip_grads', action='store_true')
    
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

       
    print('Loading config file')
    config = BertConfig.from_json_file(args.config_path)
    config_dict = config.__dict__
    

    run_name = None
    run_name = run_name if run_name is not None else make_run_name(RUN_NAME_FORMAT, phase='pretrain', config=config_dict)
    logger = make_logger(run_name, args.log_output)
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(config)

    logger.info('Loading vocab file')
    vocab = pkl.load(open(args.vocab_path, 'rb'), encoding='bytes')
    vocab_size_file = len(vocab)
    print("Vocab Size from vocab type: ", vocab_size_file)

    vocab_size = config_dict['vocab_size']
    print('Vocabulary Size from config:', vocab_size)

    logger.info('Loading training/valid dataset')
    train_dataset = pkl.load(open(args.train_dataset, 'rb'), encoding='bytes')
    val_dataset = pkl.load(open(args.val_dataset, 'rb'), encoding='bytes')

    logger.info('Creating Dataloader')
    train_dataloader = BERTdataEHRloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = BERTdataEHRloader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    

    logger.info('Building BERT model for pretraining')
    model = BertForPreTraining(config)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    loss_model = BertPretrainingCriterion(vocab_size)

    #optimizer = NoamOptimizer(model.parameters(),
                              #d_model=args.hidden_size, factor=2, warmup_steps=10000, betas=(0.9, 0.999), weight_decay=0.01)
  
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


    checkpoint_dir = make_checkpoint_dir(args.checkpoint_dir, run_name, config)

    logger.info('Start training...')
    trainer = BertTrainer(
        model = model,
        loss_model=loss_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler = scheduler,
        clip_grads=args.clip_grads,
        logger=logger,
        checkpoint_dir=args.checkpoint_dir,
        print_every=args.print_every,
        save_every=args.save_every,
        with_cuda = True
    )

    trainer.run(epochs=args.epochs)

if __name__ == "__main__":
    pretrain()
    wandb.finish()




