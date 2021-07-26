"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
import pickle
from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
import numpy as np

def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, args.gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    if args.char_embeddings:
        char_vectors = util.torch_from_json(args.char_emb_file)
    else:
        char_vectors = None


    # Get model
    log.info('Building model...')
    #model = BiDAF(word_vectors = word_vectors,
    #              char_vectors = char_vectors,
    #              hidden_size=args.hidden_size,
    #              rnn_type=args.rnn_type,
    #              self_att=args.self_att)
    model = BiDAF(word_vectors = word_vectors,
                  char_vectors = char_vectors,
                  hidden_size=args.hidden_size,
                  rnn_type=args.rnn_type,
                  self_att=args.self_att,
                  rnn_layers=[args.rnn_layers_enc,args.rnn_layers_mod,args.rnn_layers_mod2])

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, args.gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    weight_log_list = []   # Log of prediction p-values vs. y-values for start/end
    y1_log = []
    y2_log = []
    p1_log = []
    p2_log = []
    
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs,cc_idxs, qw_idxs, qc_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)
            y1_log.append(y1.cpu().detach().numpy())
            y2_log.append(y2.cpu().detach().numpy())
            p1_log.append(p1.cpu().detach().numpy())
            p2_log.append(p2.cpu().detach().numpy())

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])
            
    sub_path2 = join(args.save_dir, args.split + '_' + args.weight_log_file)
    y1_log = np.concatenate(y1_log)
    y2_log = np.concatenate(y2_log)
    maxLen = max([x.shape[1] for x in p1_log])
    X1 = [np.pad(p, 
           ((0,0),(0,maxLen - p.shape[1] )), 
           'constant',
           constant_values=0)
         for p in p1_log]
    X2 = [np.pad(p, 
           ((0,0),(0,maxLen - p.shape[1] )), 
           'constant',
           constant_values= 0)
         for p in p2_log]

    p1_log = np.concatenate(X1, axis = 0)
    p2_log = np.concatenate(X2, axis = 0)
    with open(sub_path2, 'wb') as out:
        weight_log_list = [y1_log,y2_log,p1_log,p2_log]
        pickle.dump(weight_log_list, out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(get_test_args())