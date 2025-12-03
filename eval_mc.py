from operator import gt
import os.path as osp
from unittest import result
from utils.util import load_file
import argparse
import logging
logger = logging.getLogger('VQA') 

map_name = {'d': 'Des   ', 'e': 'Exp   ', 'p': 'Pred-A', 'c': 'CF-A  ', 'pr': 'Pred-R', 'cr': 'CF-R  ', 'par':'Pred  ', 'car': 'CF    ', 'all': 'ALL   '}

# Map numeric qtype to string qtype
QTYPE_NUM_TO_STR = {
    0: 'd',   # descriptive
    1: 'e',   # explanatory
    2: 'p',   # predictive answer
    3: 'pr',  # predictive reason
    4: 'c',   # counterfactual answer
    5: 'cr',  # counterfactual reason
}

QTYPE_STR_TO_NUM = {v: k for k, v in QTYPE_NUM_TO_STR.items()}

def accuracy_metric(result_file, qtype):
    if qtype == -1:
        accuracy_metric_cvid(result_file)
    if qtype == 0:
        accuracy_metric_q0(result_file)
    if qtype == 1:
        accuracy_metric_q1(result_file)
    if qtype == 2:
        accuracy_metric_q2(result_file)
    if qtype == 3:
        accuracy_metric_q3(result_file)

def accuracy_metric_q0(result_file):
    preds = list(load_file(result_file).items())
    group_acc = {'D': 0}
    group_cnt = {'D': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)):
        id_qtypes = preds[idx]
        answer = id_qtypes[1]['answer']
        pred = id_qtypes[1]['prediction']
        group_cnt['D'] += 1
        all_cnt += 1
        if answer == pred:
            group_acc['D'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q1(result_file):
    preds = list(load_file(result_file).items())
    group_acc = {'E': 0}
    group_cnt = {'E': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)):
        id_qtypes = preds[idx]
        answer = id_qtypes[1]['answer']
        pred = id_qtypes[1]['prediction']
        group_cnt['E'] += 1
        all_cnt += 1
        if answer == pred:
            group_acc['E'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q2(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['PA', 'PR', 'P']
    group_acc = {'PA': 0, 'PR': 0, 'P': 0}
    group_cnt = {'PA': 0, 'PR': 0, 'P': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//2):
        id_qtypes = preds[idx*2:(idx+1)*2]
        qtypes = [0, 1]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(2):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['P'] += 1
        all_cnt += 1
        if answer[0] == pred[0] and answer[1] == pred[1]:
            group_acc['P'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q3(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['CA', 'CR', 'C']
    group_acc = {'CA': 0, 'CR': 0, 'C': 0}
    group_cnt = {'CA': 0, 'CR': 0, 'C': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//2):
        id_qtypes = preds[idx*2:(idx+1)*2]
        qtypes = [0, 1]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(2):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['C'] += 1
        all_cnt += 1
        if answer[0] == pred[0] and answer[1] == pred[1]:
            group_acc['C'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_all(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['D', 'E', 'PA', 'PR', 'CA', 'CR', 'P', 'C']
    group_acc = {'D': 0, 'E': 0, 'PA': 0, 'PR': 0, 'CA': 0, 'CR': 0, 'P': 0, 'C': 0}
    group_cnt = {'D': 0, 'E': 0, 'PA': 0, 'PR': 0, 'CA': 0, 'CR': 0, 'P': 0, 'C': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//6):
        id_qtypes = preds[idx*6:(idx+1)*6]
        qtypes = [int(id_qtype[0].split('_')[-1]) for id_qtype in id_qtypes]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(6):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['C'] += 1
        group_cnt['P'] += 1
        all_cnt += 4
        if answer[0] == pred[0]:
            all_acc += 1
        if answer[1] == pred[1]:
            all_acc += 1
        if answer[2] == pred[2] and answer[3] == pred[3]:
            group_acc['P'] += 1
            all_acc += 1
        if answer[4] == pred[4] and answer[5] == pred[5]:
            group_acc['C'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))


def accuracy_metric_cvid(result_file, gt_file=None, return_details=False):
    """
    Evaluate CausalVidQA results
    Now supports numeric qtype (0-5) from DataLoader
    
    qtype mapping:
        0: 'd'  - Descriptive
        1: 'e'  - Explanatory  
        2: 'p'  - Predictive Answer
        3: 'pr' - Predictive Reason
        4: 'c'  - Counterfactual Answer
        5: 'cr' - Counterfactual Reason
    
    Combined metrics:
        'par' - Predictive (answer AND reason both correct)
        'car' - Counterfactual (answer AND reason both correct)
    
    Args:
        result_file: Path to prediction JSON file
        gt_file: Not used (kept for compatibility)
        return_details: If True, return detailed results dict
    
    Returns:
        Overall accuracy (float), or tuple (accuracy, details_dict) if return_details=True
    """
    group_acc = {'d':0, 'e':0, 'p':0, 'pr':0, 'c':0, 'cr':0, 'par':0, 'car':0}
    group_cnt = {'d':0, 'e':0, 'p':0, 'pr':0, 'c':0, 'cr':0, 'par':0, 'car':0}
    
    preds = load_file(result_file)
    
    if len(preds) == 0:
        logger.warning("No predictions to evaluate!")
        return 0 if not return_details else (0, group_acc, group_cnt)
    
    # Group predictions by qtype
    qns_group = {'d':[], 'e':[], 'p':[], 'pr':[], 'c':[], 'cr':[]}
    
    for qid in preds.keys():
        # qid format: "video_id_qtype" where qtype is 0-5 (numeric)
        parts = qid.rsplit('_', 1)
        if len(parts) != 2:
            logger.warning(f"Invalid qid format: {qid}")
            continue
        try:
            qtype_num = int(parts[1])
            qtype_str = QTYPE_NUM_TO_STR.get(qtype_num)
            if qtype_str is None:
                logger.warning(f"Unknown qtype number: {qtype_num} for qid: {qid}")
                continue
            qns_group[qtype_str].append(qid)
        except ValueError:
            # Maybe qtype is already string format
            if parts[1] in qns_group:
                qns_group[parts[1]].append(qid)
            else:
                logger.warning(f"Cannot parse qtype from qid: {qid}")
    
    # Calculate accuracy for each question type
    for qtype, qids in qns_group.items():
        acc = 0
        for qid in qids:
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']
            if answer == pred: 
                acc += 1
        group_cnt[qtype] = len(qids)
        group_acc[qtype] = acc
    
    # Calculate combined accuracy for Predictive (answer + reason)
    # For each video, both p (qtype=2) and pr (qtype=3) must be correct
    acc_par = 0
    for qid in qns_group['p']:
        vid = qid.rsplit('_', 1)[0]
        reason_qid = vid + '_3'  # predictive reason
        if reason_qid in preds:
            if (preds[qid]['answer'] == preds[qid]['prediction'] and 
                preds[reason_qid]['answer'] == preds[reason_qid]['prediction']):
                acc_par += 1
    
    # Calculate combined accuracy for Counterfactual (answer + reason)
    # For each video, both c (qtype=4) and cr (qtype=5) must be correct
    acc_car = 0
    for qid in qns_group['c']:
        vid = qid.rsplit('_', 1)[0]
        reason_qid = vid + '_5'  # counterfactual reason
        if reason_qid in preds:
            if (preds[qid]['answer'] == preds[qid]['prediction'] and 
                preds[reason_qid]['answer'] == preds[reason_qid]['prediction']):
                acc_car += 1
    
    group_acc['par'] = acc_par
    group_acc['car'] = acc_car
    group_cnt['par'] = group_cnt['p']  # Same count as individual
    group_cnt['car'] = group_cnt['c']  # Same count as individual

    # Overall accuracy: Des + Exp + Pred(combined) + CF(combined)
    # This means each video contributes 4 scores: D, E, Pred(both), CF(both)
    all_acc = group_acc['d'] + group_acc['e'] + group_acc['par'] + group_acc['car']
    all_cnt = group_cnt['d'] + group_cnt['e'] + group_cnt['par'] + group_cnt['car']

    # Print results
    print("\n" + "="*70)
    print("CausalVidQA Evaluation Results")
    print("="*70)
    print(f"{'Type':<12} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-"*70)
    
    type_order = ['d', 'e', 'p', 'pr', 'c', 'cr', 'par', 'car']
    for qtype in type_order:
        acc = group_acc[qtype]
        cnt = group_cnt[qtype]
        if cnt > 0:
            pct = acc * 100.0 / cnt
            print(f"{map_name[qtype]:<12} {acc:>10} {cnt:>10} {pct:>11.2f}%")
        else:
            print(f"{map_name[qtype]:<12} {'N/A':>10} {cnt:>10} {'N/A':>12}")
    
    print("-"*70)
    if all_cnt > 0:
        overall_acc = all_acc * 100.0 / all_cnt
        print(f"{'ALL':.<12} {all_acc:>10} {all_cnt:>10} {overall_acc:>11.2f}%")
    else:
        overall_acc = 0
        print(f"{'ALL':.<12} {'N/A':>10} {all_cnt:>10} {'N/A':>12}")
    print("="*70 + "\n")
    
    # Also log for training logs
    logger.info("    ".join([map_name[q] for q in type_order]))
    results_str = []
    for qtype in type_order:
        if group_cnt[qtype] > 0:
            results_str.append('{:.2f}'.format(group_acc[qtype]*100.0/group_cnt[qtype]))
        else:
            results_str.append('N/A')
    logger.info("     ".join(results_str))
    logger.info('Overall Acc: {:.2f}%'.format(overall_acc))
    
    if return_details:
        return overall_acc, group_acc, group_cnt
    return overall_acc 



def main(result_file, mode='val'):
    print('Evaluating {}'.format(result_file))
    dataset_dir = '../data/datasets/causalvid/'
    gt_file = osp.join(dataset_dir, mode+'.csv')
    accuracy_metric_cvid(result_file, gt_file)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", type=str, default='val', choices=['val','test'])
    # parser.add_argument("--folder", type=str)
    # args = parser.parse_args()

    # result_file = f'./save_models/causalvid/{args.folder}/{args.mode}-res.json'
    # main(result_file, args.mode)
    result_file = "convert.json"  # prediction文件
    gt_file = "/storage_fast/ycli/vqa/qa_dataset/causalvid/with_qid/test.csv"  # gt_file指的就是生成这个prediction文件的数据集csv文件
    accuracy_metric_cvid(result_file, gt_file=gt_file)