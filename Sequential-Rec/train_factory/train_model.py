import time
from tqdm import tqdm
from SeqRec.eval_factory.comp_performance import rec_scores
from SeqRec.train_factory.utils import LadderSampler, gen_train_batch, gen_eval_batch, reset_random_seed
from torch.utils.data import DataLoader
from copy import deepcopy


def trainer(epoch, data_loader, model, optimizer, args):
    print('+' * 30, 'Epoch {}'.format(epoch), '+' * 30)
    start_time = time.time()
    model.train()
    running_loss = 0.0
    processed_batch = 0
    batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
    for batch_idx, (src_items, trg_items, data_size) in batch_iterator:
        optimizer.zero_grad()
        src = src_items.to(args.device)
        target = trg_items.to(args.device)
        data_size = data_size.to(args.device)
        # logits = model(src, data_size, incremental_state=None, chunkwise_recurrent=True)
        logits = model(src, data_size)
        logits = logits.view(-1, logits.size(-1))
        target = target.view(-1)
        loss = args.loss_fn(logits, target, ignore_index=0)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
        processed_batch = processed_batch + 1
        batch_iterator.set_postfix_str('Loss={:.4f}'.format(loss.item()))
    cost_time = time.time() - start_time
    avg_loss = running_loss / processed_batch
    print('Time={:.4f}, Average Loss={:.4f}'.format(cost_time, avg_loss))
    return optimizer


def train(model, optimizer, train_data, eval_data, args):
    reset_random_seed(args.seed)
    eval_loader = DataLoader(dataset=eval_data, batch_size=args.eval_batch_size,
                             num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                             collate_fn=lambda e: gen_eval_batch(e, eval_data, args.max_len))
    best_metric = rec_scores(eval_loader, model, eval_data, args)
    count = 0
    for epoch in range(1, args.num_epoch, 1):
        train_loader = DataLoader(dataset=train_data, batch_size=args.train_batch_size,
                                  sampler=LadderSampler(train_data, args.train_batch_size),
                                  num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                  collate_fn=lambda e: gen_train_batch(e, train_data, args.max_len))
        optimizer = trainer(epoch, train_loader, model, optimizer, args)
        current_metric = rec_scores(eval_loader, model, eval_data, args)

        indicator = 0
        for metric in current_metric.keys():
            if current_metric[metric] > best_metric[metric]:
                indicator = 1
        if indicator == 1:
            best_metric = current_metric
            f = open(args.result_path, 'w')
            print('Epoch:', epoch, file=f)
            for k, v in best_metric.items():
                print(k, '=', v, file=f)
            f.close()
            best_model = deepcopy(model)
            best_model.save(args.model_path)
            count = 0
        else:
            count = count + 1
        if count == 10:
            print('Early Stop!')
            break
    for k, v in best_metric.items():
        print(k, '=', v)
