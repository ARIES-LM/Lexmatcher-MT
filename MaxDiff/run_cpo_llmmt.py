#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import accelerate
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np

import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM
)

from arguments import ModelArguments, DataTrainingArguments
from trl import CPOTrainer, CPOConfig

logger = logging.getLogger(__name__)


from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CPOConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


    transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")



    ## load cpo dataset
    train_datasets = load_dataset('json', data_files=data_args.cpo_data_path, split='train')
    
    # # load tokenizer
    set_seed(training_args.seed)
    tokenizer_kwargs = {
    "use_fast_tokenizer": model_args.use_fast_tokenizer,
    "add_eos_token": False,
}
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path, **tokenizer_kwargs)

    # # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        trust_remote_code=True,
            )


    # Initialize our Trainer
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        processing_class=tokenizer
    )

    # train
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        trainer.train(resume_from_checkpoint=checkpoint)

        if model_args.use_peft:
            if torch.distributed.get_rank() == 0:
                model.save_pretrained(model_args.save_path) 
        else:
            trainer.save_model(model_args.save_path)  # Saves the tokenizer too for easy upload

    print(f"save model to {model_args.save_path}")

if __name__ == "__main__":
    main()