import torch
from tqdm import tqdm

def evaluate(args, model, dataloader, logger):
    model.eval()
    with torch.no_grad():
        image_features = []
        text_features = []
        num_anns = dataloader.dataset.num_anns
        num_ids = len(dataloader.dataset)
        num_imgs = dataloader.dataset.img_length
        for idx, batch in enumerate(dataloader):

            images, texts, img_id = batch 
            
            images = images.cuda()
            texts = texts.cuda()

            batch_image_features = model.encode_image(images)
            batch_text_features = model.encode_text(texts)

            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)

            image_features.append(batch_image_features)
            text_features.append(batch_text_features)

            if idx % args.display == 0:
                logger.info("step:%d/%d", idx, len(dataloader))

        images_ids = torch.arange(0, num_ids, num_anns).cuda()
        image_features = torch.cat(image_features, dim=0)[images_ids]
        text_features = torch.cat(text_features, dim=0)

        sim_matrix = []
        
        for idx, image_feat in tqdm(enumerate(image_features)):
            sim_line = image_feat @ text_features.t()
            sim_matrix.append(sim_line.unsqueeze(0).cpu())
        
        sim_matrix = torch.cat(sim_matrix, dim=0)
        label = torch.eye(num_imgs).unsqueeze(-1).repeat(1,1,num_anns).view(-1, num_ids)
        results = metric_compute(sim_matrix, label, logger)
    return results['mean_R1']

def evaluate_1K(args, model, dataloader, logger):
    logger.info("Testing MSCOCO 1K.")
    model.eval()
    with torch.no_grad():
        image_features = []
        text_features = []
        num_anns = dataloader.dataset.num_anns
        num_ids = len(dataloader.dataset)
        num_imgs = dataloader.dataset.img_length
        for idx, batch in enumerate(dataloader):

            images, texts, img_id = batch 
            
            images = images.cuda()
            texts = texts.cuda()

            batch_image_features = model.encode_image(images)
            batch_text_features = model.encode_text(texts)

            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)

            image_features.append(batch_image_features)
            text_features.append(batch_text_features)

            if idx % args.display == 0:
                logger.info("step:%d/%d", idx, len(dataloader))

        images_ids = torch.arange(0, num_ids, num_anns).cuda()
        image_features = torch.cat(image_features, dim=0)[images_ids]
        text_features = torch.cat(text_features, dim=0)

        sim_matrix = []
        i2t_R1 = 0
        i2t_R5 = 0
        i2t_R10 = 0
        t2i_R1 = 0 
        t2i_R5 = 0
        t2i_R10 = 0
        for i in range(5):
            img_embs = image_features[i * 1000 : (i + 1) * 1000]
            cap_embs = text_features[i * 5000 : (i + 1) * 5000]

            sim_matrix = img_embs @ cap_embs.t()
            
            label = torch.eye(1000).unsqueeze(-1).repeat(1,1,num_anns).view(-1, int(num_ids/5)).cuda()
            results = metric_compute(sim_matrix, label, logger)
            i2t_R1 = i2t_R1 + results['i2t_R@1']
            i2t_R5 = i2t_R5 + results['i2t_R@5']
            i2t_R10 = i2t_R10 + results['i2t_R@10']
            t2i_R1 = t2i_R1 + results['t2i_R@1']
            t2i_R5 = t2i_R5 + results['t2i_R@5']
            t2i_R10 = t2i_R10 + results['i2t_R@10']
        logger.info("Image-to-Text Average:")
        logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(i2t_R1/5, i2t_R5/5, i2t_R10/5))
        logger.info("Text-to-Image Average:")
        logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(t2i_R1/5, t2i_R5/5, t2i_R10/5))
        mean_R1 = (i2t_R1/5 + t2i_R1/5)/2
    return mean_R1

def metric_compute(sim_matrix, label, logger):
    results = {}
    # Image-to-Text
    i2t_rank_matrix = (-sim_matrix).argsort().argsort() + 1
    i2t_gt_rk_matrix = label * i2t_rank_matrix
    i2t_gt_rk_matrix[i2t_gt_rk_matrix==0] = 1e9
    i2t_min_rank = i2t_gt_rk_matrix.min(1).values

    results['i2t_R@1'] = 100 * torch.where(i2t_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['i2t_R@5'] = 100 * torch.where(i2t_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['i2t_R@10'] = 100 * torch.where(i2t_min_rank <= 10, 1, 0).type(torch.float32).mean()

    logger.info("Image-to-Text:")
    logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
    
    # Text-to-Image
    t2i_rank_matrix = (-sim_matrix.T).argsort().argsort() + 1
    t2i_gt_rk_matrix = label.T * t2i_rank_matrix
    t2i_gt_rk_matrix[t2i_gt_rk_matrix==0] = 1e9
    t2i_min_rank = t2i_gt_rk_matrix.min(1).values

    results['t2i_R@1'] = 100 * torch.where(t2i_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['t2i_R@5'] = 100 * torch.where(t2i_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['t2i_R@10'] = 100 * torch.where(t2i_min_rank <= 10, 1, 0).type(torch.float32).mean()

    logger.info("Text-to-Image:")
    logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
    
    results['mean_R1'] = (results['i2t_R@1'] + results['t2i_R@1']) / 2

    logger.info("Mean R1: {:.2f}".format(results['mean_R1']))
    
    return results
    