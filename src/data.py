import torch
from torch.utils.data import TensorDataset
import json
from transformers import AutoTokenizer
import re
from tqdm import tqdm


def read_raw_dataset(file_path):
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            all_data.append(json.loads(line))

    return all_data


def process_data(
    data, 
    tokenizer,
    params,
    test_mode=False,
):
    max_seq_length = params['max_seq_length']
    mode = params['model_mode']
    max_cand_length = 8

    data_dict = {
        'question_id': [],
        'answer_id': [],
        'label': [],
        'candidate': [],
        'context_mask': [],
        'context': [],
        'token_type_id': [],
    }
    if (mode == 'bi'):
        max_ctxt_length = max_seq_length
    elif (mode == 'cross'):
        max_ctxt_length = max_seq_length-4-1 # 四字成语与一个[SEP]
    else:
        print('Invalid mode')
        return None
    
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    for idx, example in tqdm(enumerate(data), desc='Process Data'):
        context = example['content']
        candidates = example['candidates']
        if not(test_mode):
            answers = example['groundTruth']
        real_count = example['realCount']

        segments = context.split('#idiom#')
        # tokenized_segments = [
        #     tokenizer.tokenize(seg) for seg in segments
        # ]
        assert len(segments) == real_count+1

        for j in range(real_count):
            if not(test_mode):
                answer = answers[j]

            candidate_list = candidates[j]
            join_str = mask_token*4
            left_context = join_str.join(segments[:j+1])
            right_context = join_str.join(segments[j+1:])
            left_tokens = tokenizer.tokenize(left_context)
            right_tokens = tokenizer.tokenize(right_context)

            left_quota = (max_ctxt_length-4-2) // 2 # -4: 成语对应的[MASK] -2: [CLS]与[SEP]
            right_quota = (max_ctxt_length-4-2) - left_quota
            if (len(left_tokens) < left_quota): # 左边token较少，右边可以多取
                if (len(right_tokens) > right_quota):
                    right_quota += (left_quota - len(left_tokens))
            elif (len(right_tokens) < right_quota): # 右边token较少，左边可以多取
                left_quota += (right_quota - len(right_tokens))

            left_tokens = left_tokens[-left_quota:]
            right_tokens = right_tokens[:right_quota]
            ctxt_tokens = left_tokens + [mask_token]*4 + right_tokens
            ctxt_tokens = [cls_token] + ctxt_tokens + [sep_token]
            
            ctxt_mask = [0]*max_seq_length
            for _tmp in range(4):
                ctxt_mask[len(left_tokens)+_tmp] = 1

            for candidate in candidate_list:
                candidate_tokens = tokenizer.tokenize(candidate)
                if not(test_mode) and (candidate == answer):
                    label = 1
                else:
                    label = 0

                data_dict['question_id'].append(idx)
                data_dict['answer_id'].append(j)
                data_dict['label'].append(label)
                data_dict['context_mask'].append(ctxt_mask)
                if (mode == 'bi'):
                    candidate_tokens = [cls_token] + candidate_tokens + [sep_token]
                    candidate_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)
                    padding = [pad_token_id] * (max_cand_length - len(candidate_ids))
                    candidate_ids += padding

                    ctxt_ids = tokenizer.convert_tokens_to_ids(ctxt_tokens)
                    padding = [pad_token_id] * (max_seq_length - len(ctxt_ids))
                    ctxt_ids += padding

                    data_dict['candidate'].append(candidate_ids)
                    data_dict['context'].append(ctxt_ids)
                    data_dict['token_type_id'].append([0]*max_seq_length)
                    
                elif (mode == 'cross'):
                    all_tokens = [cls_token] + left_tokens + candidate_tokens + right_tokens + [sep_token]
                    all_ids = tokenizer.convert_tokens_to_ids(all_tokens)
                    padding = [pad_token_id] * (max_seq_length - len(all_ids))
                    all_ids += padding

                    data_dict['candidate'].append([0]*max_cand_length)
                    data_dict['context'].append(all_ids)
    
    context_vecs = torch.tensor(data_dict['context'], dtype=torch.long)
    context_mask_vecs = torch.tensor(data_dict['context_mask'], dtype=torch.long)
    qid_vecs = torch.tensor(data_dict['question_id'], dtype=torch.long)
    aid_vecs = torch.tensor(data_dict['answer_id'], dtype=torch.long)
    label_vecs = torch.tensor(data_dict['label'], dtype=torch.long)
    candidate_vecs = torch.tensor(data_dict['candidate'], dtype=torch.long)
        
    tensor_data = TensorDataset(
        qid_vecs, 
        aid_vecs, 
        context_vecs, 
        context_mask_vecs, 
        label_vecs, 
        candidate_vecs, 
    )

    return tensor_data


if __name__ == '__main__':
    pass