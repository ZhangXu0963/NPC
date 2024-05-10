import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from clip.criterions import TotalLoss
global logger

def get_mbank(dataloader, model, noise_ratio, dataset_name):
    '''
    model: pre-trained clip or your own model
    noise_ratio: the proportion of noise in the dataset, for example, 0, 0.2, and 0.4.
    dataset_name: choose in {MSCOCO, Flickr30K, CC120K}
    '''
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        loss = []
        text_index = []
        clean_img = []
        clean_cap = []
        image_feature = []
        text_feature = []
        loss_fun = TotalLoss()
        for idx, batch in enumerate(dataloader):
            if idx%20 == 0:
                logger.info("Calculating loss for all samples: %d/%d", idx, len(dataloader))
            image, text, text_idx, img_id = batch
            
            image = image.cuda()
            text = text.cuda()
            
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)

            lce = loss_fun(logits_per_image, logits_per_text, 'mbank')
            loss.append(lce)
            text_index.append(text_idx)
            image_feature.append(image_features)
            text_feature.append(text_features)

        loss = torch.cat(loss)
        loss = loss.reshape(-1,1)

        text_index = torch.cat(text_index)
        image_feature = torch.cat(image_feature)
        text_feature = torch.cat(text_feature)
        
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        loss = loss.cpu().numpy()
        gmm.fit(loss)

        prob = gmm.predict_proba(loss)

        prob = prob[:, gmm.means_.argmin()]

        arg_c = np.argwhere((prob>0.9) == True)

        clean_idx = text_index[arg_c].cpu()

        image_feature = image_feature.cpu().numpy()
        text_feature = text_feature.cpu().numpy()

        clean_img = image_feature[clean_idx]
        clean_cap = text_feature[clean_idx]

    mbank_img_idx = {}
    mbank_txt_idx = {}

    clean_img = torch.tensor(clean_img).squeeze().cuda()
    clean_cap = torch.tensor(clean_cap).squeeze().cuda()
    clean_idx = np.array(clean_idx)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader) :
            if idx%20 == 0:
                logger.info("Calculating loss for all samples: %d/%d", idx, len(dataloader))
            images, img_ids, texts, txt_ids = batch #get data
            # texts, txt_ids = batch
            
            images = images.cuda()
            texts = texts.cuda()
            #------ clip ------
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            image_sim = image_features @ clean_img.t()
            txt_sim = text_features @ clean_cap.t()
            #---------------------------------------------------------------------------------------------
            img_max_sim, img_max_idx = torch.topk(image_sim, k=1, dim=1, largest = True)
            txt_max_sim, txt_max_idx = torch.topk(txt_sim, k=1, dim=1, largest = True)
            #---------------------------------------------------------------------------------------------
            img_max_idx = img_max_idx.cpu()
            txt_max_idx = txt_max_idx.cpu()
            m_img_idx = clean_idx[img_max_idx]
            m_txt_idx = clean_idx[txt_max_idx]
            #------------------------------------------------------------------------------------------
            for i in range(len(img_ids)):
                mbank_img_idx[str(int(img_ids[i]))] = m_img_idx[i]

            for i in range(len(txt_ids)):
                mbank_txt_idx[str(int(txt_ids[i]))] = m_txt_idx[i]
        #----------------------------------------------------------------------------------------------------------------------------
        np.save('dataset/{}/annotations/query_bank/{}_mbank_img_idx.npy'.format(dataset_name, str(noise_ratio)), mbank_img_idx)
        np.save('dataset/{}/annotations/query_bank/{}_mbank_txt_idx.npy'.format(dataset_name, str(noise_ratio)), mbank_txt_idx)








