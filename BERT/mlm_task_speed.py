import argparse
import time
import math
from adaptdl.torch import frozen_layer_count
import torch
import torch.nn as nn
from model import MLMTask
from utils import run_demo, run_ddp, wrap_up
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter  # Added for tensorboard
from torch.utils.data import DataLoader
import adaptdl  # Changed in step 1
import adaptdl.torch  # Changed in step 1
from adaptdl.torch import init_process_group, profile_layer_info, AdaptiveDataParallel 
from adaptdl.torch.utils.misc import isAtomicLayer, apply_frozen
import os  # Added for tensorboard 
import numpy as np 
from adaptdl.torch._metrics import _metrics_state, get_progress

from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

import collections
import random
import copy 
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)



def collect_atomic_layer_num(net):
    atomic_layer_num = 0
    for module in net.modules():
        if isAtomicLayer(module):
            atomic_layer_num += 1
    return atomic_layer_num


def PipeTransformerMethod(layer_num, alpha, epoch):
    second_term = 0.0
    for e in range(2, epoch + 1):
        second_term += ((layer_num * alpha) / pow(1 - alpha, e))
    return pow(1 - alpha, epoch) * ((layer_num * alpha) / (1 - alpha) + second_term)


def cal_frozen_layer(epoch, tot_epoch, tot_layer_num):
    fronzen_layer_num = int(PipeTransformerMethod(tot_layer_num, 1.0 / tot_epoch, epoch)) // 2 * 2 
    return fronzen_layer_num


def frozen_net(net):
    for module in net.modules():
        if isAtomicLayer(module):
            frozen = True
            for param in module.parameters():
                if param.requires_grad == True:
                    frozen = False
            if frozen:
                module.eval() 




def collate_batch(batch_data, args, mask_id, cls_id): 
    if len(batch_data) % args.bptt != 0:
        # print(len(batch_data))
        batch_data = batch_data[:len(batch_data)//args.bptt*args.bptt]
    batch_data = \
        torch.tensor(batch_data).long().view(args.bptt, -1).t().contiguous()
    # Generate masks with args.mask_frac
    data_len = batch_data.size(0)
    ones_num = int(data_len * args.mask_frac)
    zeros_num = data_len - ones_num
    lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
    lm_mask = lm_mask[torch.randperm(data_len)]
    
    batch_data = \
        torch.cat((torch.tensor([[cls_id] * batch_data.size(1)]).long(),
                  batch_data))
    lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

    targets = torch.stack(
        [batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, lm_mask, targets


def process_raw_data(raw_data, args):
    _num = raw_data.size(0) // (args.batch_size * args.bptt)
    raw_data = raw_data[:(_num * args.batch_size * args.bptt)]
    return raw_data


def evaluate(data_source, model, vocab, ntokens, criterion, args, device,
             test=False, epoch=None, writer=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if test:
        total_loss = 0.
    else:
        stats = adaptdl.torch.Accumulator()

    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    dataloader = DataLoader(
        data_source, batch_size=args.batch_size * args.bptt,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id))
    with torch.no_grad():
        for batch, (data, lm_mask, targets) in enumerate(dataloader):
            if args.parallel == 'DDP':
                data = data.to(device[0])
                targets = targets.to(device[0])
            else:
                data = data.to(device)
                targets = targets.to(device)
            data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel
            output = model([data, None])
            output = torch.stack(
                [output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
            output_flat = output.view(-1, ntokens)
            if test:
                total_loss += criterion(output_flat, targets).item()
            else:
                stats['test_loss'] += criterion(output_flat, targets).item()
                stats['total'] += targets.size(0)

    if test:
        return total_loss / ((len(data_source) - 1) / args.bptt / 32)

    with stats.synchronized():
        test_loss = (stats['test_loss'] / ((len(data_source) - 1) / args.bptt / args.batch_size) /
                     adaptdl.env.num_replicas())
        writer.add_scalar("Loss/valid", test_loss, epoch) 
        print('ppl {}'.format(math.exp(test_loss)))

    return test_loss


def train(model, vocab, train_loss_log, train_data,
          optimizer, criterion, ntokens, epoch, scheduler,
          args, device, rank=None, batch_size_log=None, writer=None, speed_total_batch_size=0, limit=20):
    # TODO: reduce number of args for this function
    model.train()
    frozen_net(model) # key step 
    total_loss = 0
    start_time = time.time()
    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    train_loss_log.append(0.0)
    base_bsz = speed_total_batch_size

    # watch out 
    from adaptdl.torch.data import AdaptiveDataLoaderHelper
    AdaptiveDataLoaderHelper._training = None 
    dataloader = adaptdl.torch.AdaptiveDataLoader(
        train_data, drop_last=True, batch_size=base_bsz, shuffle=False,
        collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id))
    dataloader._elastic.train()  
    set_local_batch_size(local_batch_size=speed_total_batch_size) 
    # dataloader.autoscale_batch_size(
    #     128 * base_bsz,
    #     local_bsz_bounds=(max(base_bsz / 4, 2048), min(2 * base_bsz, 4096*2)),
    #     gradient_accumulation=args.gradient_accumulation)

    for batch, (data, lm_mask, targets) in enumerate(dataloader):
        if batch >= limit:
            break   
        if batch == 0: 
            print(data.shape)
        optimizer.zero_grad() 
        if args.parallel == 'DDP':
            print("DDP")
            data = data.to(device[0])
            targets = targets.to(device[0])
        else:
            data = data.to(device)
            targets = targets.to(device)
        data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel 
        output = model([data, None])
        output = torch.stack(
            [output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        # if not model.scaling_rule.is_accumulation_step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            batch = batch // (dataloader.accumulation_steps + 1)
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            if (rank is None) or rank == 0:
                train_loss_log[-1] = cur_loss
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                      'ms/batch {:5.2f} | loss {:5.2f} | '
                      'ppl {:8.2f} | batch_size {:5d}'.format(
                          epoch, batch,
                          len(train_data) // dataloader.current_batch_size,
                          scheduler.get_last_lr()[0],
                          elapsed * 1000 / args.log_interval,
                          cur_loss, math.exp(cur_loss),
                          dataloader.current_batch_size))
            total_loss = 0
            start_time = time.time() 
    return dataloader

def walkover(trainloader): 
    for idx, data in enumerate(trainloader):  
        pass


def run_main(args, rank=None):
    torch.manual_seed(args.seed)
    if args.parallel == 'DDP':
        n = torch.cuda.device_count() // args.world_size
        device = list(range(rank * n, (rank + 1) * n))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    import torchtext 
    import adaptdl 
    import adaptdl.torch 

    if args.dataset == 'WikiText103':
        from experimental.datasets import WikiText103 as WLMDataset
    elif args.dataset == 'WikiText2':
        from experimental.datasets import WikiText2 as WLMDataset
    elif args.dataset == 'WMTNewsCrawl':
        from data import WMTNewsCrawl as WLMDataset
    elif args.dataset == 'EnWik9':
        from torchtext.datasets import EnWik9
    elif args.dataset == 'BookCorpus':
        from data import BookCorpus
    else:
        print("dataset for MLM task is not supported")

    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, test_dataset, valid_dataset = WLMDataset()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)

    if args.dataset == 'WikiText103' or args.dataset == 'WikiText2':
        train_dataset, test_dataset, valid_dataset = WLMDataset(vocab=vocab)
    elif args.dataset == 'WMTNewsCrawl':
        from torchtext.experimental.datasets import WikiText2
        test_dataset, valid_dataset = WikiText2(vocab=vocab, data_select=('test', 'valid'))
        train_dataset, = WLMDataset(vocab=vocab, data_select='train')
    elif args.dataset == 'EnWik9':
        enwik9 = EnWik9()
        idx1, idx2 = int(len(enwik9) * 0.8), int(len(enwik9) * 0.9)
        train_data = torch.tensor([vocab.stoi[_id]
                                  for _id in enwik9[0:idx1]]).long()
        val_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx1:idx2]]).long()
        test_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx2:]]).long()
        from torchtext.experimental.datasets import LanguageModelingDataset
        train_dataset = LanguageModelingDataset(train_data, vocab)
        valid_dataset = LanguageModelingDataset(val_data, vocab)
        test_dataset = LanguageModelingDataset(test_data, vocab)
    elif args.dataset == 'BookCorpus':
        train_dataset, test_dataset, valid_dataset = BookCorpus(vocab)


    train_data = process_raw_data(train_dataset.data, args)
    # if rank is not None:
    #     # Chunk training data by rank for different gpus
    #     chunk_len = len(train_data) // args.world_size
    #     train_data = train_data[(rank * chunk_len):((rank + 1) * chunk_len)]
    val_data = process_raw_data(valid_dataset.data, args)
    test_data = process_raw_data(test_dataset.data, args)

    ntokens = len(train_dataset.get_vocab())
    if args.checkpoint != 'None':
        model = torch.load(args.checkpoint)
    else:
        # print(ntokens)
        # print(args.emsize)
        # print(args.nhead)
        # print(args.nhid)
        # print(args.nlayers)
        # print(args.dropout)
        model = MLMTask(
            ntokens, args.emsize, args.nhead,
            args.nhid, args.nlayers, args.dropout) 
        # exit(0)
    if args.parallel == 'DDP':
        model = model.to(device[0])
        model = DDP(model, device_ids=device)
    else:
        model = model.to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{"params": [param]} for param in model.parameters()], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    # lr_scheduler = ExponentialLR(optimizer, 1.0 ** (1.0 / args.epochs))
    # import pdb; pdb.set_trace() 
    init_process_group("nccl" if torch.cuda.is_available() else "gloo") 
    profile_layer_info(model, [torch.arange(33).unsqueeze(0).long(), None])
    adl_model = AdaptiveDataParallel(model, optimizer, lr_scheduler=scheduler, patch_optimizer=True, find_unused_parameters=True)
    
    tot_layer_num = collect_atomic_layer_num(adl_model)
    profile_tot_layer = int(tot_layer_num * 0.7)

    best_val_loss = None
    train_loss_log, val_loss_log, batch_size_log = [], [], []

    tensorboard_dir = os.path.join(
        os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")
        if adaptdl.env.replica_rank() == 0 else "/tmp",
        adaptdl.env.job_id())

    writer = SummaryWriter(tensorboard_dir)
    
    frozen_list = np.linspace(0, profile_tot_layer, 10)
    frozen_list = [int(layer) for layer in frozen_list] 

    for epoch, frozen_layer_num in enumerate(frozen_list): 
        adaptdl.torch.set_current_frozen_layer(frozen_layer_num)
        apply_frozen(adl_model, frozen_layer_num)
        for local_batch_size in [8, 16, 24, 32, 48, 64]: 
            torch.cuda.reset_accumulated_memory_stats(0)
            torch.cuda.reset_peak_memory_stats(0)

            metric_info = _metrics_state() 
            info_dict = dict() 
            metric_info.profile = collections.defaultdict(collections.Counter) 
            path = 'speed/model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(args.dataset, args.placement, local_batch_size, args.placement, frozen_layer_num)
            metric_list = list() 
            # if os.path.exists(path): 
            #     continue 
            trainloader = train(adl_model, train_dataset.vocab, train_loss_log, train_data,
                optimizer, criterion, ntokens, epoch, scheduler, args,
                device, rank, batch_size_log, writer, speed_total_batch_size=local_batch_size*128)


            metric_list.append(copy.deepcopy(_metrics_state())) 
            walkover(trainloader) 
            info_dict['metric_{}'.format(frozen_layer_num)] = metric_list 
            print(metric_list[0].profile)
            if adaptdl.env.replica_rank() == 0: 
                print('local_batch_size == {}, frozen layer {}, memory {}'.format(local_batch_size, frozen_layer_num, torch.cuda.max_memory_allocated(0))) 
                # print('frozen layer: {}'.format(frozen_layer_num))
                with open('speed/model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(args.dataset, args.placement, local_batch_size, args.placement, frozen_layer_num), 'wb') as f:
                    np.save(f, info_dict) 

def set_local_batch_size(local_batch_size): 
    batch_size = local_batch_size * adaptdl.env.num_replicas() 
    os.environ['TARGET_BATCH_SIZE'] = "{}".format(batch_size) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--emsize', type=int, default=768,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=3072,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=12,
                        help='the number of heads in the encoder/decoder of'
                             'the transformer model')
    parser.add_argument('--lr', type=float, default=6,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=5431916812,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='None',
                        help='path to load the checkpoint')
    parser.add_argument('--save', type=str, default='mlm_bert.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str,
                        default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    parser.add_argument('--dataset', type=str, default='WikiText2',
                        help='dataset used for MLM task')
    parser.add_argument('--parallel', type=str, default='None',
                        help='Use DataParallel to train model')
    parser.add_argument('--world_size', type=int, default=8,
                        help='the world size to initiate DPP')
    parser.add_argument('--gradient-accumulation',
                        dest='gradient_accumulation',
                        default=False, action='store_true',
                        help='Enable gradient accumulation')
    parser.add_argument('--frozen', dest='frozen', default=False,
                    action='store_true', help='autoscale batchsize')
    parser.add_argument('--placement', required=True, type=str, help='placement')
    args = parser.parse_args()

    if args.parallel == 'DDP':
        run_demo(run_ddp, run_main, args)
    else:
        run_main(args, adaptdl.env.replica_rank()) 
