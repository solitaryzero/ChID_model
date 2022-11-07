# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target, reduction='mean'):
        loss = -(target * torch.pow(torch.sigmoid(-input), self.gamma) * F.logsigmoid(input) + 
                (1 - target) * torch.pow(torch.sigmoid(input), self.gamma) * F.logsigmoid(-input))
        class_dim = input.dim() - 1
        C = input.size(class_dim)
        loss = loss.sum(dim=class_dim) / C
        if reduction == "none":
            ret = loss
        elif reduction == "mean":
            ret = loss.mean()
        elif reduction == "sum":
            ret = loss.sum()
        else:
            ret = input
            raise ValueError(reduction + " is not valid")
        return ret


class BertEncoder(nn.Module):
    def __init__(self, bert_model):
        super(BertEncoder, self).__init__()
        bert_output_dim = bert_model.config.hidden_size

        self.bert_model = bert_model
        self.out_dim = bert_output_dim

    def forward(self, token_ids, segment_ids, attention_mask, token_mask=None):
        outputs = self.bert_model(
            input_ids=token_ids, 
            token_type_ids=segment_ids, 
            attention_mask=attention_mask
        )

        add_mask = token_mask is not None

        # get embedding of [CLS] token
        if add_mask:
            embeddings = outputs.last_hidden_state
        else:
            # embeddings = outputs.pooler_output
            embeddings = outputs.last_hidden_state[:,0,:]

        if add_mask:
            assert token_mask is not None
            embeddings = embeddings * token_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1)
            embeddings = embeddings / token_mask.sum(-1, keepdim=True)
            result = embeddings
        else:
            result = embeddings

        return result


class BiEncoderModel(nn.Module):
    def __init__(self, params):
        super(BiEncoderModel, self).__init__()
        # init device
        self.params = params
        self.device = torch.device(
            "cuda" if params["cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        # init structure
        ctxt_bert = AutoModel.from_pretrained(params['base_model'])
        cand_bert = AutoModel.from_pretrained(params['base_model'])
        self.context_encoder = BertEncoder(ctxt_bert)
        self.cand_encoder = BertEncoder(cand_bert)
        self.config = ctxt_bert.config

        # load params
        if (params['load_model_path'] is not None):
            self.load_model(params['load_model_path'])

        # init tokenizer
        self.NULL_IDX = 0
        self.tokenizer = AutoTokenizer.from_pretrained(params["base_model"])


    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def to_bert_input(self, token_idx, null_idx):
        """ token_idx is a 2D tensor int.
            return token_idx, segment_idx and mask
        """
        segment_idx = token_idx * 0
        mask = (token_idx != null_idx)
        # nullify elements in case self.NULL_IDX was not 0
        token_idx = token_idx * mask.long()
        return token_idx, segment_idx, mask


    def encode(
        self,
        context_input,
        cand_input,
        context_mask=None,
        candidate_mask=None,
    ):
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            context_input, self.NULL_IDX
        )
        token_idx_cands, segment_idx_cands, mask_cands = self.to_bert_input(
            cand_input, self.NULL_IDX
        )

        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt, context_mask
            )
            # embedding_ctxt = F.normalize(embedding_ctxt)

        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands, candidate_mask
            )
            # embedding_cands = F.normalize(embedding_cands)
        
        return embedding_ctxt, embedding_cands


    def encode_context(self, context_input, context_mask=None):
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            context_input, self.NULL_IDX
        )

        embedding_ctxt = self.context_encoder(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, context_mask
        )

        return embedding_ctxt

    
    def encode_candidate(self, cand_input, candidate_mask=None):
        token_idx_cands, segment_idx_cands, mask_cands = self.to_bert_input(
            cand_input, self.NULL_IDX
        )

        embedding_cands = self.cand_encoder(
            token_idx_cands, segment_idx_cands, mask_cands, candidate_mask
        )

        return embedding_cands


    def score(
        self,
        context_input,
        cand_input,
        context_mask,
        candidate_mask,
    ):
        embedding_ctxt, embedding_cands = self.encode(
            context_input=context_input,
            cand_input=cand_input,
            context_mask=context_mask,
            candidate_mask=candidate_mask,
        )

        semantic_scores = embedding_ctxt.mm(embedding_cands.t())
        # semantic_scores = F.softmax(semantic_scores, dim=1)
        semantic_scores = torch.sigmoid(semantic_scores)

        semantic_scores = semantic_scores.cpu().detach().numpy()

        return semantic_scores


    def score_candidates(
        self,
        context_input,
        cand_inputs,
        context_mask=None,
    ):
        embedding_ctxt = self.encode_context(context_input, context_mask)
        assert cand_inputs.shape[0] == 1
        cand_inputs = cand_inputs.squeeze(0)
        embedding_cands = self.encode_candidate(cand_inputs)

        semantic_scores = embedding_ctxt.mm(embedding_cands.t())
        # semantic_scores = torch.sigmoid(semantic_scores)
        # semantic_scores = F.softmax(semantic_scores, dim=1)

        return semantic_scores

    def score(
        self,
        context_input,
        cand_input,
        context_mask=None,
    ):
        embedding_ctxt = self.encode_context(context_input, context_mask)
        embedding_cand = self.encode_candidate(cand_input)
        semantic_scores = (embedding_ctxt * embedding_cand).sum(dim=1)
        # semantic_scores = F.cosine_similarity(embedding_ctxt, embedding_cand) * 0.5 + 0.5

        return semantic_scores


    def forward(
        self, 
        context_input, 
        cand_input, 
        context_mask=None,
        label=None,
    ):
        assert (label is not None)
        semantic_scores = self.score(
            context_input=context_input,
            cand_input=cand_input,
            context_mask=context_mask,
        )
        # sem_loss = F.binary_cross_entropy_with_logits(semantic_scores, label.float())
        sem_loss = torch.abs(label.float()-semantic_scores)

        return sem_loss, semantic_scores


class CrossEncoderModel(nn.Module):
    def __init__(self, params):
        super(CrossEncoderModel, self).__init__()
        # init device
        self.params = params
        self.device = torch.device(
            "cuda" if params["cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        # init structure
        cross_bert = AutoModel.from_pretrained(params['base_model'])
        self.encoder = BertEncoder(cross_bert)
        self.dropout = nn.Dropout(0.1)

        self.score_layer = nn.Sequential(
            self.dropout,
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(self.encoder.out_dim, 1)
        )
        

        self.config = cross_bert.config
        
        # load params
        if (params['load_model_path'] is not None):
            self.load_model(params['load_model_path'])

        # init tokenizer
        self.NULL_IDX = 0
        self.tokenizer = AutoTokenizer.from_pretrained(params["base_model"])


    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def to_bert_input(self, token_idx, null_idx):
        """ token_idx is a 2D tensor int.
            return token_idx, segment_idx and mask
        """
        segment_idx = token_idx * 0
        mask = (token_idx != null_idx)
        # nullify elements in case self.NULL_IDX was not 0
        token_idx = token_idx * mask.long()
        return token_idx, segment_idx, mask


    # def to_bert_input(self, token_idx, null_idx):
    #     """ token_idx is a 2D tensor int.
    #         return token_idx, segment_idx and mask
    #     """
    #     mask = token_idx != null_idx
    #     # nullify elements in case self.NULL_IDX was not 0
    #     token_idx = token_idx * mask.long()
    #     return token_idx, mask


    def encode(
        self,
        context_input,
    ):
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = self.to_bert_input(
            context_input, self.NULL_IDX
        )

        embedding_ctxt = self.encoder(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
        )
        
        return embedding_ctxt


    def score(
        self,
        context_input,
    ):
        embedding_ctxt = self.encode(
            context_input=context_input,
        )
        semantic_scores = self.score_layer(embedding_ctxt).squeeze(dim=1)

        return semantic_scores


    def forward(
        self, 
        context_input, 
        label,
    ):
        semantic_scores = self.score(
            context_input=context_input,
        )
        sem_loss = F.binary_cross_entropy_with_logits(semantic_scores, label.float())
        # sem_loss = F.mse_loss(semantic_scores, label.float())

        return sem_loss, semantic_scores