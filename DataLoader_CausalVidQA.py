import torch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
import nltk
import h5py
import os
import json
import pickle as pkl
import re
from numpy import random

def pkload(file):
    """Load pickle file"""
    data = None
    if osp.exists(file) and osp.getsize(file) > 0:
        with open(file, 'rb') as fp:
            data = pkl.load(fp)
    return data

def pkdump(data, file):
    """Save pickle file"""
    dirname = osp.dirname(file)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(file, 'wb') as fp:
        pkl.dump(data, fp)


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, word2idx=None, idx2word=None):
        self.word2idx = word2idx if word2idx else {}
        self.idx2word = idx2word if idx2word else {}
        self.idx = len(self.idx2word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx.get('<UNK>', 0)
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class CausalVidQADataset(Dataset):
    """
    DataLoader for CausalVidQA dataset
    
    Args:
        feature_path: Path to visual features (appearance_feat.h5, motion_feat.h5, ROI_text.h5)
        text_feature_path: Path to text features (text_seq.h5, bert_adj_dict.pkl, etc.)
        split_path: Path to split file (train.pkl, val.pkl, test.pkl)
        data_path: Path to text annotations (text.json, answer.json for each video)
        use_bert: Whether to use BERT features
        vocab: Vocabulary object (required if use_bert=False)
        qtype: Question type (-1 for all, 0-5 for specific type)
        max_length: Maximum sequence length
    """

    def __init__(self, feature_path, text_feature_path, split_path, data_path, 
                 use_bert=True, vocab=None, qtype=-1, max_length=40):
        self.feature_path = feature_path
        self.text_feature_path = text_feature_path
        self.split_path = split_path
        self.data_path = data_path
        self.qtype = qtype
        self.vocab = vocab
        self.max_length = max_length
        self.use_bert = use_bert

        # Load video ids for this split
        self.vids = pkload(self.split_path)
        print(f"Loaded {len(self.vids)} videos from {split_path}")

        # Load BERT features if using BERT
        if self.use_bert:
            self.bert_file = osp.join(text_feature_path, 'text_seq.h5')
            self.bert_length = osp.join(text_feature_path, 'text_seq_length.pkl')
            self.bert_token = osp.join(text_feature_path, 'token_org.pkl')
            
            with open(self.bert_token, 'rb') as fbt:
                self.token_dict = pkl.load(fbt)
            with open(self.bert_length, 'rb') as fl:
                self.length_dict = pkl.load(fl)

        # Load adjacency matrix for dependency parsing
        if self.use_bert:
            self.adj_path = osp.join(text_feature_path, 'bert_adj_dict.pkl')
        else:
            self.adj_path = osp.join(text_feature_path, 'glove_adj_dict.pkl')
        
        if osp.exists(self.adj_path):
            with open(self.adj_path, 'rb') as fbt:
                self.token_adj = pkl.load(fbt)
        else:
            self.token_adj = None
            print(f"Warning: Adjacency file not found at {self.adj_path}")

        # Load video feature index mapping
        vf_info = pkload(osp.join(feature_path, 'idx2vid.pkl'))
        self.vf_info = dict()
        for idx, vid in enumerate(vf_info):
            if vid in self.vids:
                self.vf_info[vid] = idx

        # Load appearance features
        app_file = osp.join(feature_path, 'appearance_feat.h5')
        print(f'Loading {app_file}...')
        self.app_feats = dict()
        with h5py.File(app_file, 'r') as fp:
            feats = fp['resnet_features']
            for vid, idx in self.vf_info.items():
                self.app_feats[vid] = feats[idx][...]

        # Load motion features
        mot_file = osp.join(feature_path, 'motion_feat.h5')
        print(f'Loading {mot_file}...')
        self.mot_feats = dict()
        with h5py.File(mot_file, 'r') as fp:
            feats = fp['resnet_features']
            for vid, idx in self.vf_info.items():
                self.mot_feats[vid] = feats[idx][...]

        # Load ROI text object features
        self.txt_obj = dict()
        roi_file = osp.join(self.feature_path, 'ROI_text.h5')
        if osp.exists(roi_file):
            print(f'Loading {roi_file}...')
            with h5py.File(roi_file, 'r') as f:
                keys = [item for item in self.vids if item in f.keys()]
                for key in keys:
                    tmp = dict()
                    labels = f[key].keys()
                    for label in labels:
                        new_label = '[' + label + ']'
                        tmp[new_label] = f[key][label][...]
                    self.txt_obj[key] = tmp
        else:
            print(f"Warning: ROI_text.h5 not found at {roi_file}")

    def __len__(self):
        if self.qtype == -1:
            return len(self.vids) * 6
        elif self.qtype == 0 or self.qtype == 1:
            return len(self.vids)
        elif self.qtype == 2 or self.qtype == 3:
            return len(self.vids) * 2
        return len(self.vids)

    def get_video_feature(self, video_name):
        """Get appearance and motion features for a video"""
        app_feat = self.app_feats[video_name]
        mot_feat = self.mot_feats[video_name]
        return (torch.from_numpy(app_feat).type(torch.float32), 
                torch.from_numpy(mot_feat).type(torch.float32))

    def get_word_idx(self, text):
        """Convert text to word indices"""
        tokens = nltk.tokenize.word_tokenize(str(text).lower())
        token_ids = [self.vocab(token) for i, token in enumerate(tokens) if i < (self.max_length - 2)]
        return token_ids

    def get_token_seq(self, text):
        """Tokenize text"""
        tokens = nltk.tokenize.word_tokenize(str(text).lower())
        return tokens

    def get_adj(self, vidx, qtype):
        """Get adjacency matrices for dependency relations"""
        if self.token_adj is None:
            # Return zero matrices if no adjacency data
            qas_adj_new = np.zeros((5, self.max_length, self.max_length))
            ques_adj_new = np.zeros((self.max_length, self.max_length))
            return qas_adj_new, ques_adj_new
            
        adj_vidx = self.token_adj[vidx]
        qas_adj = adj_vidx[6 + qtype * 5:11 + qtype * 5]
        ques_adj = adj_vidx[qtype]
        
        qas_adj_new = np.zeros((len(qas_adj), self.max_length, self.max_length))
        ques_adj_new = np.zeros((self.max_length, self.max_length))
        
        for idx, item in enumerate(qas_adj):
            if item.shape[0] > self.max_length:
                qas_adj_new[idx] = item[:self.max_length, :self.max_length]
            else:
                qas_adj_new[idx, :item.shape[0], :item.shape[1]] = item
                
        if ques_adj.shape[0] > self.max_length:
            ques_adj_new = ques_adj[:self.max_length, :self.max_length]
        else:
            ques_adj_new[:ques_adj.shape[0], :ques_adj.shape[1]] = ques_adj
            
        return qas_adj_new, ques_adj_new

    def get_trans_matrix(self, candidates):
        """Convert candidate answers to matrix form"""
        qa_lengths = [len(qa) for qa in candidates]
        candidates_matrix = torch.zeros([5, self.max_length]).long()
        
        for k in range(5):
            sentence = candidates[k]
            length = qa_lengths[k]
            if length > self.max_length:
                length = self.max_length
                candidates_matrix[k] = torch.Tensor(sentence[:length])
            else:
                candidates_matrix[k, :length] = torch.Tensor(sentence)

        return candidates_matrix, qa_lengths

    def get_ques_matrix(self, ques):
        """Convert question to matrix form"""
        q_lengths = len(ques)
        ques_matrix = torch.zeros([self.max_length]).long()
        ques_matrix[:q_lengths] = torch.Tensor(ques)
        return ques_matrix, q_lengths

    def get_tagname(self, line):
        """Extract tag names from text"""
        tag = set()
        tmp_tag = re.findall(r"\[(.+?)\]", line)
        for item in tmp_tag:
            tag.add('[' + item + ']')
        return list(tag)

    def match_tok_tag(self, labels, tags, tok):
        """Match tokens with tags"""
        tok_tag = [None for _ in range(len(tok))]
        if labels == list():
            return tok_tag
            
        for tag in tags:
            for idx in range(len(tok)):
                if tag.startswith(tok[idx]):
                    new_idx = idx
                    while new_idx < len(tok) and not tag.endswith(tok[new_idx]):
                        new_idx += 1
                    if new_idx < len(tok):
                        new_tag = ''.join(tok[idx:new_idx + 1])
                        if new_tag == tag:
                            for i in range(idx, new_idx + 1):
                                tok_tag[i] = tag
                            if tag not in labels:
                                label = random.choice(labels)
                                for index, item in enumerate(tok_tag):
                                    if item == tag:
                                        tok_tag[index] = label
        return tok_tag

    def load_txt_obj(self, vid, tok, org):
        """Load text-aligned object features"""
        if vid in self.txt_obj:
            labels = list(self.txt_obj[vid].keys())
        else:
            labels = list()
            
        fea = list()
        for idx in range(len(tok)):
            tags = self.get_tagname(org[idx])
            tok_tag = self.match_tok_tag(labels, tags, tok[idx])
            fea_each = list()
            
            for item in tok_tag:
                if item is None:
                    fea_each.append(np.zeros((2048,)))
                else:
                    fea_each.append(self.txt_obj[vid][item])
            fea_each = np.stack(fea_each, axis=0)
            
            new_fea_each = np.zeros((self.max_length, 2048))
            if fea_each.shape[0] > self.max_length:
                new_fea_each = fea_each[:self.max_length]
            else:
                new_fea_each[:fea_each.shape[0]] = fea_each

            fea.append(new_fea_each)
        return fea

    def load_text(self, vid, qtype):
        """Load question, candidate answers, and ground truth answer"""
        text_file = os.path.join(self.data_path, vid, 'text.json')
        answer_file = os.path.join(self.data_path, vid, 'answer.json')
        
        with open(text_file, 'r') as fin:
            text = json.load(fin)
        with open(answer_file, 'r') as fin:
            answer = json.load(fin)

        # Question types:
        # 0: descriptive
        # 1: explanatory  
        # 2: predictive answer
        # 3: predictive reason
        # 4: counterfactual answer
        # 5: counterfactual reason
        if qtype == 0:
            qns = text['descriptive']['question']
            cand_ans = text['descriptive']['answer']
            ans_id = answer['descriptive']['answer']
        elif qtype == 1:
            qns = text['explanatory']['question']
            cand_ans = text['explanatory']['answer']
            ans_id = answer['explanatory']['answer']
        elif qtype == 2:
            qns = text['predictive']['question']
            cand_ans = text['predictive']['answer']
            ans_id = answer['predictive']['answer']
        elif qtype == 3:
            qns = text['predictive']['question']
            cand_ans = text['predictive']['reason']
            ans_id = answer['predictive']['reason']
        elif qtype == 4:
            qns = text['counterfactual']['question']
            cand_ans = text['counterfactual']['answer']
            ans_id = answer['counterfactual']['answer']
        elif qtype == 5:
            qns = text['counterfactual']['question']
            cand_ans = text['counterfactual']['reason']
            ans_id = answer['counterfactual']['reason']
        else:
            raise ValueError(f"Invalid qtype: {qtype}")
            
        return qns, cand_ans, ans_id

    def load_text_bert(self, vid, qtype):
        """Load BERT text features"""
        try:
            import SharedArray as sa
            feature = sa.attach(f"shm://{vid}")
        except:
            # Fallback: load from h5 file directly
            with h5py.File(self.bert_file, 'r') as fp:
                feature = fp[vid][...]
                
        token_org = self.token_dict[vid]
        length = self.length_dict[vid]
        
        cand = feature[6 + qtype * 5:11 + qtype * 5]
        tok = token_org[0][6 + qtype * 5:11 + qtype * 5]
        org = token_org[1][6 + qtype * 5:11 + qtype * 5]
        cand_l = length[6 + qtype * 5:11 + qtype * 5]
        
        question = feature[qtype]
        tok_q = [token_org[0][qtype], ]
        org_q = [token_org[1][qtype], ]
        qns_len = length[qtype]

        dim = cand.shape[2]
        new_candidate = np.zeros((5, self.max_length, dim))
        new_question = np.zeros((self.max_length, dim))
        
        for idx, qa_l in enumerate(cand_l):
            if qa_l > self.max_length:
                new_candidate[idx] = cand[idx, :self.max_length]
            else:
                new_candidate[idx, :qa_l] = cand[idx, :qa_l]
                
        if qns_len > self.max_length:
            new_question = question[:self.max_length]
        else:
            new_question[:qns_len] = question[:qns_len]

        return (torch.from_numpy(new_candidate).type(torch.float32), tok, org, cand_l,
                torch.from_numpy(new_question).type(torch.float32), tok_q, org_q, qns_len)

    def __getitem__(self, idx):
        # Determine question type based on index
        if self.qtype == -1:
            qtype = idx % 6
            idx = idx // 6
        elif self.qtype == 0 or self.qtype == 1:
            qtype = self.qtype
        elif self.qtype == 2:
            qtype = 2 + (idx % 2)
            idx = idx // 2
        elif self.qtype == 3:
            qtype = 4 + (idx % 2)
            idx = idx // 2
        else:
            qtype = self.qtype

        vidx = self.vids[idx]

        # Load text data
        qns, cand_ans, ans_id = self.load_text(vidx, qtype)

        if self.use_bert:
            candidate, tok, org, can_lengths, question, tok_q, org_q, qns_len = self.load_text_bert(vidx, qtype)
        else:
            tok_q = [['<START>', ] + self.get_token_seq(qns) + ['<END>', ], ]
            org_q = [qns, ]
            question, qns_len = self.get_ques_matrix(
                [self.vocab('<START>'), ] + self.get_word_idx(qns) + [self.vocab('<END>'), ])

            tok = []
            org = []
            candidate = []
            qnstok = ['<START>', ] + self.get_token_seq(qns) + ['<END>', ]
            qnsids = [self.vocab('<START>'), ] + self.get_word_idx(qns) + [self.vocab('<END>'), ]
            
            for ans in cand_ans:
                anstok = ['<START>', ] + self.get_token_seq(ans) + ['<END>', ]
                ansids = [self.vocab('<START>'), ] + self.get_word_idx(ans) + [self.vocab('<END>'), ]
                tok.append(qnstok + anstok)
                org.append(qns + ans)
                candidate.append(qnsids + ansids)
                
            candidate, can_lengths = self.get_trans_matrix(candidate)

        can_lengths = torch.tensor(can_lengths).clamp(max=self.max_length)
        qns_len = torch.tensor(qns_len).clamp(max=self.max_length)

        # Load object features aligned with text
        obj_feature = torch.from_numpy(
            np.stack(self.load_txt_obj(vidx, tok, org), axis=0)).type(torch.float32)
        obj_feature_q = torch.from_numpy(
            self.load_txt_obj(vidx, tok_q, org_q)[0]).type(torch.float32)

        # Load dependency relation adjacency matrices
        adj_qas, adj_ques = self.get_adj(vidx, qtype)
        adj_ques = torch.from_numpy(adj_ques).type(torch.float32)
        adj_qas = torch.from_numpy(np.stack(adj_qas, axis=0)).type(torch.float32)

        # Load video features
        app_feature, mot_feature = self.get_video_feature(vidx)

        qns_key = vidx + '_' + str(qtype)

        return ([app_feature, mot_feature], 
                [candidate, can_lengths, obj_feature, adj_qas], 
                [question, qns_len, obj_feature_q, adj_ques], 
                torch.tensor(ans_id), 
                qns_key)


def create_causalvid_dataloader(split, feature_path, text_feature_path, split_dir, 
                                 data_path, use_bert=True, vocab=None, qtype=-1, 
                                 max_length=40, batch_size=32, shuffle=True, num_workers=4):
    """
    Helper function to create CausalVidQA DataLoader
    
    Args:
        split: 'train', 'val', or 'test'
        feature_path: Path to visual features
        text_feature_path: Path to text features
        split_dir: Directory containing split files
        data_path: Path to text annotations
        use_bert: Whether to use BERT features
        vocab: Vocabulary (required if use_bert=False)
        qtype: Question type (-1 for all)
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader instance
    """
    split_path = osp.join(split_dir, f'{split}.pkl')
    
    dataset = CausalVidQADataset(
        feature_path=feature_path,
        text_feature_path=text_feature_path,
        split_path=split_path,
        data_path=data_path,
        use_bert=use_bert,
        vocab=vocab,
        qtype=qtype,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    import argparse
    
    # Example usage with kagglehub paths
    # import kagglehub
    # text_feature_path = kagglehub.dataset_download('lusnaw/text-feature')
    # visual_feature_path = kagglehub.dataset_download('lusnaw/visual-feature')
    # split_path = kagglehub.dataset_download('lusnaw/dataset-split-1')
    # text_annotation_path = kagglehub.dataset_download('lusnaw/text-annotation')
    
    parser = argparse.ArgumentParser(description="CausalVidQA DataLoader Test")
    parser.add_argument('--feature_path', type=str, required=True, help='Path to visual features')
    parser.add_argument('--text_feature_path', type=str, required=True, help='Path to text features')
    parser.add_argument('--split_path', type=str, required=True, help='Path to split file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to text annotations')
    parser.add_argument('--use_bert', action='store_true', default=True)
    parser.add_argument('--qtype', type=int, default=-1, help='Question type (-1 for all)')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    dataset = CausalVidQADataset(
        feature_path=args.feature_path,
        text_feature_path=args.text_feature_path,
        split_path=args.split_path,
        data_path=args.data_path,
        use_bert=args.use_bert,
        qtype=args.qtype
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    for sample in dataloader:
        video_feats, candidate_info, question_info, ans_id, qns_key = sample
        app_feat, mot_feat = video_feats
        candidate, can_lengths, obj_feature, adj_qas = candidate_info
        question, qns_len, obj_feature_q, adj_ques = question_info
        
        print("=" * 50)
        print("Appearance feature shape:", app_feat.shape)
        print("Motion feature shape:", mot_feat.shape)
        print("Candidate shape:", candidate.shape)
        print("Candidate lengths:", can_lengths)
        print("Object feature (candidates) shape:", obj_feature.shape)
        print("Adjacency QAS shape:", adj_qas.shape)
        print("Question shape:", question.shape)
        print("Question length:", qns_len)
        print("Object feature (question) shape:", obj_feature_q.shape)
        print("Adjacency question shape:", adj_ques.shape)
        print("Answer ID:", ans_id)
        print("Question key:", qns_key)
        print("=" * 50)
        break

    print("DataLoader test completed!")
