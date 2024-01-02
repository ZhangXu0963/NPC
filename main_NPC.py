import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import clip
from clip.criterions import TotalLoss
from torch import optim
from util import set_seed_logger, get_logger
from params import parse_args
from scheduler import cosine_lr
from eval import evaluate, evaluate_1K
from data import prepare_dataloader
import copy
from torch import autograd

global logger

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def set_requires_grad(net: nn.Module, mode=True):
    for p in net.parameters():
        p.requires_grad_(mode)

def main():
    global logger
    args = parse_args()

    seed = set_seed_logger(args)
    dir_path = os.path.join(args.checkpoint_path, args.experiments)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    logger = get_logger(os.path.join(dir_path, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model_clip, preprocess = clip.clip.load(args.vision_model, device=device, jit=False) #Must set jit=False for training
    siamese_model_clip, siamese_preprocess = clip.clip.load(args.vision_model, device=device, jit=False) #Must set jit=False for training

    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model_clip
        siamese_model = siamese_model_clip
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))


    else:
        model = model_clip
        siamese_model = siamese_model_clip
        logger.info("Model Initialized!")

    model = model.cuda()
    siamese_model = siamese_model.cuda()    

    if args.eval:
        train_dataloader = None
        train_length = 0
        args.epochs = 0
        dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'test')
        test_dataloader, test_length = dataloader['test']
        if args.dataset == 'coco':
            eval_Rank = evaluate(args, model, test_dataloader, logger)
            eval_Rank_1K = evaluate_1K(args, model, test_dataloader, logger) # Only for MSCOCO 1K testing
        else:
            eval_Rank = evaluate(args, model, test_dataloader, logger)
    else:
        dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'train')
        train_dataloader, train_length = dataloader['train']
        dataloader_dev = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'dev')
        dev_dataloader, dev_length = dataloader_dev['dev']

    loss = TotalLoss()
    loss = loss.cuda()

    total_steps = train_length * args.epochs

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) 

    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    siamese_optimizer = optim.AdamW(siamese_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    siamese_scheduler = cosine_lr(siamese_optimizer, args.lr, args.warmup, total_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    best_score = 0
    
    for epoch in range(args.epochs):

        # NPC training
        train(args, model, siamese_model, train_dataloader, train_length, epoch, scheduler, siamese_scheduler, optimizer, siamese_optimizer, loss)
        
        save_path = os.path.join(dir_path, f"epoch{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )
        logger.info("Saved checkpoint {} (epoch {})".format(save_path, epoch + 1))

        ## Run on val dataset for selecting best model.
        logger.info("Eval on val dataset")
        eval_Rank = evaluate(args, model, dev_dataloader, logger)

        if best_score <= eval_Rank:
            best_score = eval_Rank
            best_output_model_file = save_path
        logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

def train(args, model, siamese_model, train_dataloader, train_length, epoch, scheduler, siamese_scheduler, optimizer, siamese_optimizer, loss):

    model.train()
    siamese_model.train()
    sloss = 0
    smbank_loss = 0

    for idx, batch in enumerate(train_dataloader) :
        step = train_length * epoch + idx
        scheduler(step)
        siamese_scheduler(step)

        images, texts, _, img_ids, mbankI_imgs, mbankI_txts, mbankT_txts, mbankT_imgs = batch

        images = images.cuda()
        texts = texts.cuda()

        mbankI_imgs = mbankI_imgs.cuda()
        mbankI_txts = mbankI_txts.cuda()

        mbankT_txts = mbankT_txts.cuda()
        mbankT_imgs = mbankT_imgs.cuda()

        for name, param in model.state_dict().items():
            siamese_model.state_dict()[name].copy_(param)

        with torch.no_grad():
            siamese_model.eval()
            mbankI_i2t_sim, mbankI_t2i_sim = siamese_model(mbankI_imgs, mbankI_txts)
            mbankT_i2t_sim, mbankT_t2i_sim = siamese_model(mbankT_imgs, mbankT_txts)

            mbankI_ori_loss = loss(mbankI_i2t_sim, mbankI_t2i_sim, 'mbank')
            mbankT_ori_loss = loss(mbankT_i2t_sim, mbankT_t2i_sim, 'mbank')
            mbank_ori_loss = mbankI_ori_loss + mbankT_ori_loss
            smbank_loss = smbank_loss + torch.mean(mbank_ori_loss)

        batch_i2t_sim2, batch_t2i_sim2 = siamese_model(images, texts)
        batch_i2t_sim, batch_t2i_sim = model(images, texts)

        batch_i2t_ori_loss, batch_t2i_ori_loss = loss(batch_i2t_sim, batch_t2i_sim, 'batch')
        batch_i2t_ori_loss2, batch_t2i_ori_loss2 = loss(batch_i2t_sim2, batch_t2i_sim2, 'batch')
        total_loss = batch_i2t_ori_loss2 + batch_t2i_ori_loss2

        siamese_optimizer.zero_grad()
        set_requires_grad(model, False)
        total_loss.backward(retain_graph = True)
        total_loss.backward()
        siamese_optimizer.step()
        set_requires_grad(model, True)

        with torch.no_grad():
            siamese_model.eval()
            mbankI_i2t_sim_new, mbankI_t2i_sim_new = siamese_model(mbankI_imgs, mbankI_txts)
            mbankT_i2t_sim_new, mbankT_t2i_sim_new = siamese_model(mbankT_imgs, mbankT_txts)
            mbank_img_new_loss = loss(mbankI_i2t_sim_new, mbankI_t2i_sim_new, 'mbank')
            mbank_txt_new_loss = loss(mbankT_i2t_sim_new, mbankT_t2i_sim_new, 'mbank')

        weight = (mbankI_ori_loss/mbank_img_new_loss + mbankT_ori_loss/mbank_txt_new_loss)/2 
        wei = torch.where(weight<1, torch.tanh(weight), 1)
        batch_loss = torch.mean(wei * (batch_i2t_ori_loss + batch_t2i_ori_loss)/2)
        final_loss = torch.mean(mbank_ori_loss) + batch_loss
        sloss = sloss + final_loss

        optimizer.zero_grad()
        set_requires_grad(siamese_model, False)
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        set_requires_grad(siamese_model, True)

        if (idx % args.display == 0) and (idx != 0):
            logger.info("Epoch: %d/%d, step:%d/%d, lr: %.8f, sloss: %f, smbank_loss: %f", 
                        epoch + 1, args.epochs, idx, len(train_dataloader), optimizer.param_groups[0]['lr'],
                        sloss/args.display, smbank_loss/args.display)
            sloss = 0
            smbank_loss = 0
    return

if __name__ == '__main__':
    main()

