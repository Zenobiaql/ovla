"""
A simple script for finetuning openvla in incremental learning setting on several simpler datasets with DDP.
"""

import os
import draccus
import tqdm
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.utils.data.distributed import DistributedSampler

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

from dataclasses import dataclass
from collections import deque

from openvla_simpler_dataset import PizzaDataset
import random

# DDP process group setup
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=torch.distributed.timedelta(seconds=3600))

# Logger setup
def get_logger(name, file_path):
    if file_path is not None:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if file_path is not None: 
        logger.addHandler(f_handler)
    
    return logger

# Finetuning configuration, passed with .yaml or .yml file
@dataclass
class FinetuneConfig:
    
    # model path
    vla_path: str

    "Data Configuration"
    # dataset root directory       
    pizza_dir: str
    # name of dataset class                        
    dataset_name: str
    # directory for log and checkpoints                            
    run_root_dir: str
    # directory for adapter checkpoints                               
    adapter_tmp_dir: str                                      

    "Finetuning Configuration"
    epochs: int                                                
    batch_size: int                                                                                        
    save_steps: int                                            
    learning_rate: float                                       
    grad_accumulation_steps: int                               
    image_aug: bool                                         
                        
    "LoRA Arguments"
    use_lora: bool                                        
    lora_rank: int                                           
    lora_dropout: float                                                                                      
                                                                      
# Model training class, adapted for different task and log/file settings    
class ModelTrain:
    def __init__(
        self,

        # training loop settings 
        epochs,
        batch_size,
        grad_accumulation_steps,

        # log/file/directory settings
        run_dir,
        adapter_dir,
        logger,
        val_logger,

        # model training settings
        optimizer, 
        vla,
        vla_path,
        processor,
        action_tokenizer,
        use_lora,

        # data loader settings
        dataloader,
        val_dataloader_set,
        task_id,
        
        # device settings
        device_id,
        
    ):
        self.epochs = epochs
        self.optimizer = optimizer
        self.vla = vla
        self.dataloader = dataloader
        self.val_dataloader_set = val_dataloader_set
        self.grad_accumulation_steps = grad_accumulation_steps
        self.action_tokenizer = action_tokenizer
        self.logger = logger
        self.val_logger = val_logger
        self.use_lora = use_lora
        self.processor = processor
        self.run_dir = run_dir
        self.adapter_dir = adapter_dir
        self.task_id = task_id
        self.vla_path = vla_path
        self.batch_size = batch_size
        self.device_id = device_id

    def _average_training_loss(self, local_loss):
        loss_tensor = torch.tensor(local_loss, dtype=torch.float32).to(self.device_id)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
        
        return avg_loss
    
    def _average_validation_loss(self, local_loss):
        loss_tensor = torch.tensor(local_loss, dtype=torch.float32).to(self.device_id)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
        
        return avg_loss
    
    # val_dataloader_set should be a dictionary of dataloaders for each trained task(including current task)
    def _validate_step(self, num_epoch):
        self.vla.eval()
        
        with torch.no_grad():
            for val_dataset_name, val_dataloader in self.val_dataloader_set.items():
                if dist.get_rank() == 0:
                    self.val_logger.info("")
                    self.val_logger.info(f"Validation after epoch{num_epoch}:")
                    print(f"Validation after epoch{num_epoch}:")
                    
                total_loss = 0
                total_accuracy = 0
                total_l1_loss = 0
                    
                for batch in tqdm.tqdm(val_dataloader, total=len(val_dataloader), desc=f"{val_dataset_name}"):
                    output: CausalLMOutputWithPast = self.vla(
                        input_ids=batch["input_ids"].to(self.device_id),
                        attention_mask=batch["attention_mask"].to(self.device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss
                    total_loss += loss.item()
                    
                    action_logits = output.logits[:, self.vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                    action_preds = action_logits.argmax(dim=2)
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    mask = action_gt > self.action_tokenizer.action_token_begin_idx
                    correct_preds = (action_preds == action_gt) & mask
                    action_accuracy = correct_preds.sum().float() / mask.sum().float()
                    total_accuracy += action_accuracy.item()

                    continuous_actions_pred = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    )
                    continuous_actions_gt = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                    )
                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                    total_l1_loss += action_l1_loss.item()
                    
                avg_loss = total_loss / len(val_dataloader)
                avg_accuracy = total_accuracy / len(val_dataloader)
                avg_action_l1_loss = total_l1_loss / len(val_dataloader)
                
                mul_avg_loss = self._average_validation_loss(avg_loss)
                mul_avg_accuracy = self._average_validation_loss(avg_accuracy)
                mul_avg_action_l1_loss = self._average_validation_loss(avg_action_l1_loss)
                
                if dist.get_rank() == 0:
                    print(f"On dataset {val_dataset_name}, Loss:{mul_avg_loss:.4f}, Accuracy:{mul_avg_accuracy:.4f}, L1 Loss:{mul_avg_action_l1_loss:.4f}.")
                    self.val_logger.info(f"On dataset {val_dataset_name}, Loss:{mul_avg_loss:.4f}, Accuracy:{mul_avg_accuracy:.4f}, L1 Loss:{mul_avg_action_l1_loss:.4f}.")
            
            if dist.get_rank() == 0:
                self.val_logger.info("Finished")


    def train_step(self):
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.vla.train()
            self.optimizer.zero_grad()

            for batch_idx, batch in tqdm.tqdm(self.dataloader, total=len(self.dataloader)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = self.vla(
                        input_ids=batch["input_ids"].to(self.device_id),
                        attention_mask=batch["attention_mask"].to(self.device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()

                action_logits = output.logits[:, self.vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > self.action_tokenizer.action_token_begin_idx

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                continuous_actions_pred = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                recent_losses = deque(maxlen=self.grad_accumulation_steps)
                recent_action_accuracies = deque(maxlen=self.grad_accumulation_steps)
                recent_l1_losses = deque(maxlen=self.grad_accumulation_steps)

                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                gradient_step_idx = batch_idx // self.grad_accumulation_steps

                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
                
                mul_smoothened_loss = self._average_training_loss(smoothened_loss)
                mul_smoothened_action_accuracy = self._average_training_loss(smoothened_action_accuracy)
                mul_smoothened_l1_loss = self._average_training_loss(smoothened_l1_loss)

                if batch_idx % 10 == 0:
                    if dist.get_rank() == 0:
                        self.logger.info(f"train_loss: {mul_smoothened_loss:.4f}, action_accuracy: {mul_smoothened_action_accuracy:.4f}, l1_loss: {mul_smoothened_l1_loss:.4f}, step: {gradient_step_idx}")

                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self._validate_step(epoch)

            if self.use_lora:
                if self.device_id == 0:
                    self.logger.info(f"Saving Model Checkpoint for epoch {epoch}")
                    self.val_logger.info("")
                    self.val_logger.info(f"Model Checkpoint for epoch {epoch} saved.")
                    save_dir = self.adapter_dir
                    self.processor.save_pretrained(os.path.join(save_dir, f'epoch{epoch}'))
                    self.vla.module.save_pretrained(os.path.join(save_dir, f'epoch{epoch}'))
#                    base_vla = AutoModelForVision2Seq.from_pretrained(
#                        self.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
#                    )
#                    merged_vla = PeftModel.from_pretrained(base_vla, os.path.join(save_dir, f'epoch{epoch}'))
#                    merged_vla = merged_vla.merge_and_unload()
#                
#                    model_param_dir = os.path.join(self.run_dir, "merged", f"epoch{epoch}")
#                    os.makedirs(model_param_dir, exist_ok=True)
#                    self.processor.save_pretrained(model_param_dir)
#                    merged_vla.save_pretrained(model_param_dir)

            else:
                if self.device_id == 0:
                    self.logger.info(f"Saving Model Checkpoint for epoch {epoch}")
                    self.val_logger.info("")
                    self.val_logger.info(f"Model Checkpoint for epoch {epoch} saved.")
                    save_dir = os.path.join(self.run_dir, "fullfinetune")
                    os.makedirs(save_dir, exist_ok=True)
                    self.processor.save_pretrained(os.path.join(save_dir, f'epoch{epoch}'))
                    self.vla.module.save_pretrained(os.path.join(save_dir, f'epoch{epoch}'))

    def update_vla(self):
        new_vla = self.vla

        return new_vla        
                

@draccus.wrap()
def finetune(cfg: FinetuneConfig)->None:
    ddp_setup()

    if dist.get_rank() == 0:
        os.makedirs(cfg.run_root_dir, exist_ok=True)

    current_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    log_path = os.path.join(cfg.run_root_dir, f"time{current_time}.log")
    val_log_path = os.path.join(cfg.run_root_dir, f"time{current_time}-validation.log")
    logger = get_logger("log", log_path)
    val_logger = get_logger("val_log", val_log_path)

    if dist.get_rank() == 0:
        logger.info(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
        logger.info(f"Training setting batch size {cfg.batch_size}, learning rate {cfg.learning_rate}")
        print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
        print(f"Training setting batch size {cfg.batch_size}, learning rate {cfg.learning_rate}")

    exp_id = (
        f"{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"

    run_dir, adapter_dir = os.path.join(cfg.run_root_dir, exp_id), os.path.join(cfg.adapter_tmp_dir, exp_id)
    if dist.get_rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    device_id = int(os.environ["LOCAL_RANK"])

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
        
    vla = vla.to(device_id)
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True)

    trainable_params = [param for param in vla.module.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    
    val_dataloader_set = {}

    data_root_dir = Path(cfg.pizza_dir)
    for task in tqdm.tqdm(data_root_dir.iterdir(), desc="Incremental Training"):
            
            if task.is_dir():
                
                task_run_dir= os.path.join(run_dir, f"task-{task.name}") 
                task_adapter_dir = os.path.join(adapter_dir, f"task-{task.name}")
                if dist.get_rank() == 0:
                    os.makedirs(task_run_dir, exist_ok=True)
                    os.makedirs(task_adapter_dir, exist_ok=True)

                # current task dataset
                task_data = PizzaDataset(
                    str(task),
                    action_tokenizer,
                    processor.tokenizer,
                    processor.image_processor.apply_transform,
                    prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
                )
                
                # divide dataset into train and validation set
                indices = list(range(len(task_data)))
                random.shuffle(indices)
                
                train_ratio = 0.8
                train_size = int(train_ratio * len(indices))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                dataloader = DataLoader(
                    Subset(task_data, train_indices),
                    batch_size=cfg.batch_size,
                    sampler=DistributedSampler(Subset(task_data, train_indices)),
                    collate_fn=collator,
                    num_workers=4,
                )
                
                val_dataloader = DataLoader(
                    Subset(task_data, val_indices),
                    batch_size=cfg.batch_size,
                    sampler=DistributedSampler(Subset(task_data, val_indices)),
                    collate_fn=collator,
                    num_workers=4,
                )
                
                # add validation dataloader of current task to the set
                val_dataloader_set[task.name] = val_dataloader
    
                model_train = ModelTrain(
                    cfg.epochs, 
                    cfg.batch_size,
                    cfg.grad_accumulation_steps,

                    task_run_dir,
                    task_adapter_dir,
                    logger,
                    val_logger,

                    optimizer, 
                    vla,
                    cfg.vla_path,
                    processor,
                    action_tokenizer,
                    cfg.use_lora,

                    dataloader,
                    val_dataloader_set,
                    task.name,
                    
                    device_id,
                )

                model_train.train_step()
                vla = model_train.update_vla()
                
            else:
                pass

    dist.destroy_process_group()
    print(f"Finished running code on rank {device_id}.")

if __name__ == "__main__":
    finetune()
    
    