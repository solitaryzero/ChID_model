import logging
import sys
import os
import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup


def setup_logger(name, save_dir, filename="log.txt", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_model(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, 'checkpoint.bin')
    torch.save(model.state_dict(), output_model_file)
    tokenizer.save_vocabulary(output_dir)


def ellipse(lst, max_display=5, sep='|'):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)


def get_optimizer(model, params):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=params['learning_rate'], 
    )

    return optimizer


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params['train_batch_size']
    epochs = params['epoch']

    num_train_steps = int(len_train_data / batch_size) * epochs
    num_warmup_steps = int(num_train_steps * params['warmup_proportion'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_train_steps,
    )
    if (logger):
        logger.info("Num optimization steps = %d" % num_train_steps)
        logger.info("Num warmup steps = %d", num_warmup_steps)
    return scheduler


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_accuracy(out, labels):
    outputs = sigmoid(out)
    outputs = np.greater_equal(outputs, 0.5)

    # print(out.shape)
    # print(out)
    # print(outputs.shape)
    # print(outputs)
    # print(labels.shape)
    # print(labels)
    # print(outputs == labels)
    # input()

    return np.sum(outputs == labels), outputs == labels