import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from model import BiEncoderModel, CrossEncoderModel
import data
import utils


def load_model(params):
    mode = params['model_mode']
    if (mode == 'bi'):
        return BiEncoderModel(params)
    elif (mode == 'cross'):
        return CrossEncoderModel(params)


def predict(
    model, eval_dataloader, params, device, logger,
):
    with torch.no_grad():
        model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = {}
        temp_scores = {}
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            question_id, answer_id, context_ids, context_masks, labels, candidates = batch
            
            if (params['model_mode'] == 'bi'):
                scores = model.score(
                    context_input=context_ids, 
                    cand_input=candidates, 
                    context_mask=context_masks,
                )
            elif (params['model_mode'] == 'cross'):
                scores = model.score(
                    context_input=context_ids, 
                )

            for i in range(question_id.shape[0]):
                qid = int(question_id[i].item())
                if (qid not in temp_scores):
                    temp_scores[qid] = {}

                aid = int(answer_id[i].item())
                if (aid not in temp_scores[qid]):
                    temp_scores[qid][aid] = []

                temp_scores[qid][aid].append(scores[i].item())


    for qid in temp_scores:
        if (qid not in results):
            results[qid] = {}
        for aid in temp_scores[qid]:
            p = -1
            max_score = -1e9
            for i, sc in enumerate(temp_scores[qid][aid]):
                if (sc > max_score):
                    max_score = sc
                    p = i

            results[qid][aid] = p
            
    return results


def main(params):
    model_path = params['model_path']
    config_path = os.path.join(model_path, 'config.json')
    load_model_path = os.path.join(model_path, 'checkpoint.bin')
    data_path = params['data_path']
    data_split = params['data_split']

    with open(config_path, 'r', encoding='utf-8') as fin:
        params = json.load(fin)
    params['load_model_path'] = load_model_path
    params['data_path'] = data_path
    params['data_split'] = data_split

    logger = utils.setup_logger('Chid', params['output_path'])

    # Init model
    model = load_model(params)
    tokenizer = model.tokenizer
    device = model.device

    model = model.to(model.device)

    eval_batch_size = params["eval_batch_size"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load eval data
    test_samples = data.read_raw_dataset(os.path.join(params["data_path"], "%s_data.json" %params['data_split']))
    test_tensor_data = data.process_data(
        test_samples,
        tokenizer,
        params,
    )

    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=eval_batch_size
    )

    results = predict(
        model, test_dataloader, params, device=device, logger=logger,
    )

    correct, total = 0.0, 0.0

    for qid, example in enumerate(test_samples):
        for aid in range(example['realCount']):
            prediction_idx = results[qid][aid]
            prediction = example['candidates'][aid][prediction_idx]
            if (prediction == example['groundTruth'][aid]):
                correct += 1

            total += 1

    out_path = os.path.join(model_path, '%s_result.txt' %(data_split))
    acc_str = 'Accuracy on %s set: %d/%d = %f' %(params['data_split'], int(correct), int(total), correct/total)
    print(acc_str)
    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write(acc_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/ChID')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--data_split', type=str, default='test', choices=['dev', 'test'])
    
    # model arguments
    parser.add_argument('--model_mode', type=str, default='bi', choices=['bi', 'cross'])
    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)