import torch
import numpy as np
import os
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
# from dataset import CustomDataset

import sys
sys.path.append('../')
from probtorch.util import grid2gif, mkdirs, apply_poe, transform
import probtorch
from sklearn.metrics import f1_score
import random

ATTR_TO_IX_DICT = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18,
                   'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38,
                   'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21,
                   'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26,
                   'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32,
                   'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22,
                   'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3,
                   'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23,
                   'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}
ATTR_IX_TO_KEEP = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
IX_TO_ATTR_DICT = {v: k for k, v in ATTR_TO_IX_DICT.items()}


def save_traverse(iters, data_loader, enc, dec, cuda, output_dir_trvsl, flatten_pixel=None, fixed_idxs=[3, 2, 1, 30, 4, 23, 21, 41, 84, 99]):

    output_dir_trvsl = '../output/' + output_dir_trvsl
    tr_range = 2
    out_dir = os.path.join(output_dir_trvsl, str(iters) +'_'+ str(-tr_range) + '~' + str(tr_range))

    fixed_XA = [0] * len(fixed_idxs)
    label = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], label[i] = data_loader.dataset.__getitem__(idx)[:2]
        if flatten_pixel:
            fixed_XA[i] = fixed_XA[i].view(-1, flatten_pixel)
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)

    fixed_XA = torch.stack(fixed_XA, dim=0)
    # cnt = [0] * 10
    # for l in label:
    #     cnt[l] += 1
    # print('cnt of digit:')
    # print(cnt)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zA_ori, zS_ori = q['privateA'].dist.loc, q['sharedA'].value

    zA_dim = zA_ori.shape[2]
    zS_dim = zS_ori.shape[2]
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, zS_dim))

    tempA = []  # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
    loc=-1
    for row in range(zA_dim):
        if loc != -1 and row != loc:
            continue
        zA = zA_ori.clone()

        temp = []
        for val in interpolation:
            zA[:, :, row] = val
            sampleA = dec.forward2(zA, zS_ori, cuda)
            if flatten_pixel:
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                sampleA = torch.transpose(sampleA, 0,1)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    tempS = []
    for i in range(zS_dim):
        zS = np.zeros((1, 1, zS_dim))
        zS[0, 0, i % zS_dim] = 1.
        zS = torch.Tensor(zS)
        zS = torch.cat([zS] * len(fixed_idxs), dim=1)
        if cuda:
            zS = zS.cuda()
        sampleA = dec.forward2(zA_ori, zS, cuda)
        if flatten_pixel:
            sampleA = sampleA.view(-1, sampleA.shape[0], 28, 28)
        tempS.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

    tempA.append(torch.cat(tempS, dim=0).unsqueeze(0))
    gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

    # save the generated files, also the animated gifs

    mkdirs(output_dir_trvsl)
    mkdirs(out_dir)

    for j, val in enumerate(interpolation):
        # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
        I = gifs[:, j]
        save_image(
            tensor=I.cpu(),
            filename=os.path.join(out_dir, '%03d.jpg' % (j)),
            nrow=1 + zA_dim + 1 + 1 + 1 + zS_dim,
            pad_value=1)
        # make animated gif
    grid2gif(
        out_dir, str(os.path.join(out_dir, 'traverse' + '.gif')), delay=10
    )

    # digit trv imgs
    ### orginal img ###
    digit_trv = []
    if flatten_pixel is not None:
        fixed_XA = fixed_XA.view(-1, 1, 28, 28)
    digit_trv.append((torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0))
    sampleA = dec.forward2(zA_ori, zS_ori, cuda)
    if flatten_pixel is not None:
        sampleA = sampleA.view(-1, 1, 28, 28)
    digit_trv.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
    WS = torch.ones(digit_trv[0].shape)
    # WS = torch.ones((sampleA.shape[0], 1, 1,1))
    if cuda:
        WS = WS.cuda()
    digit_trv.append(WS)
    digit_trv.extend(tempS)
    mkdirs(out_dir + '/digit/')
    save_image(
        tensor=(torch.cat(digit_trv, dim=0)).cpu(),
        filename=os.path.join(out_dir + '/digit/', 'digit.jpg'),
        nrow=3 + zS_dim,
        pad_value=1)


def resize(h,w,img, cuda):
    from torchvision import transforms
    import torchvision
    resized_imgs = []
    for i in range(img.shape[0]):
        img_PIL = transforms.ToPILImage()(img[i].data.detach().cpu())
        img_PIL = torchvision.transforms.Resize([h,w])(img_PIL)
        resized_imgs.append(torch.transpose(torchvision.transforms.ToTensor()(img_PIL),1,2))
    resized_imgs = torch.stack(resized_imgs)
    if cuda:
        resized_imgs = resized_imgs.cuda()
    return resized_imgs


def save_traverse_both(iters, data_loader, encA, decA, encB, decB, cuda, output_dir_trvsl, flatten_pixel=None, fixed_idxs = [3246, 7001, 14305, 19000, 27444, 33100, 38000, 45231, 51000, 55121]):

    tr_range = 2
    out_dir = os.path.join('../output/' + output_dir_trvsl, str(iters) +'_'+ str(-tr_range) + '~' + str(tr_range))

    fixed_XA = [0] * len(fixed_idxs)
    fixed_XB = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):

        fixed_XA[i], fixed_XB[i] = \
            data_loader.dataset.__getitem__(idx)[0:2]
        fixed_XB[i] = fixed_XB[i].view(-1, flatten_pixel)
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
            fixed_XB[i] = fixed_XB[i].cuda()
        fixed_XB[i] = fixed_XB[i].squeeze(0)


    fixed_XA = torch.stack(fixed_XA, dim=0)
    fixed_XB = torch.stack(fixed_XB, dim=0)

    # do traversal and collect generated images

    q = encA(fixed_XA, num_samples=1)
    q = encB(fixed_XB, num_samples=1, q=q)

    zA_ori, zSA_ori = q['privateA'].dist.loc, q['sharedA'].value
    zB_ori, zSB_ori = q['privateB'].dist.loc, q['sharedB'].value

    # making poe dist
    prior_logit = torch.zeros_like(q['sharedA'].dist.logits)  # prior is the concrete dist. of uniform dist.
    poe_logit = q['sharedA'].dist.logits + q['sharedB'].dist.logits + prior_logit
    q.concrete(logits=poe_logit,
               temperature=0.66,
               name='poe')
    # sampling poe
    zS_ori = q['poe'].value

    zA_dim = zA_ori.shape[2]
    zB_dim = zB_ori.shape[2]
    zS_dim = zS_ori.shape[2]
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, 10))



    #### A private
    tempAll = []  # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32

    loc=-1
    for row in range(zA_dim):
        if loc != -1 and row != loc:
            continue
        zA = zA_ori.clone()

        temp = []
        for val in interpolation:
            zA[:, :, row] = val
            sampleA = decA.forward2(zA, zS_ori, cuda)
            sampleA = resize(28,28,sampleA,cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    # shared A
    tempS = []
    for i in range(zS_dim):
        zS = np.zeros((1, 1, zS_dim))
        zS[0, 0, i % zS_dim] = 1.
        zS = torch.Tensor(zS)
        zS = torch.cat([zS] * len(fixed_idxs), dim=1)
        if cuda:
            zS = zS.cuda()
        sampleA = decA.forward2(zA_ori, zS, cuda)
        sampleA = resize(28, 28, sampleA, cuda)
        tempS.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
    tempAll.append(torch.cat(tempS, dim=0).unsqueeze(0))

    tempAll2 = []  # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
    ###### B_private
    for row in range(zB_dim):
        if loc != -1 and row != loc:
            continue
        zB = zB_ori.clone()

        temp = []
        for val in interpolation:
            zB[:, :, row] = val
            sampleB = decB.forward2(zB, zS_ori, cuda)
            sampleB = sampleB.view(sampleB.shape[0], -1, 28, 28)
            sampleB = torch.transpose(sampleB, 0, 1)
            sampleB_3ch = []
            for i in range(sampleB.size(0)):
                each_XB = sampleB[i].clone().squeeze(0)
                sampleB_3ch.append(torch.stack([each_XB, each_XB, each_XB]))
            sampleB_3ch = torch.stack(sampleB_3ch)
            temp.append((torch.cat([sampleB_3ch[i] for i in range(sampleB_3ch.shape[0])], dim=1)).unsqueeze(0))
        tempAll2.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    # shared B
    tempS = []
    for i in range(zS_dim):
        zS = np.zeros((1, 1, zS_dim))
        zS[0, 0, i % zS_dim] = 1.
        zS = torch.Tensor(zS)
        zS = torch.cat([zS] * len(fixed_idxs), dim=1)
        if cuda:
            zS = zS.cuda()
        sampleB = decB.forward2(zB_ori, zS, cuda)
        sampleB = sampleB.view(sampleB.shape[0], -1, 28, 28)
        sampleB = torch.transpose(sampleB, 0, 1)
        sampleB_3ch = []
        for i in range(sampleB.size(0)):
            each_XB = sampleB[i].clone().squeeze(0)
            sampleB_3ch.append(torch.stack([each_XB, each_XB, each_XB]))
        sampleB_3ch = torch.stack(sampleB_3ch)
        tempS.append((torch.cat([sampleB_3ch[i] for i in range(sampleB_3ch.shape[0])], dim=1)).unsqueeze(0))
    tempAll2.append(torch.cat(tempS, dim=0).unsqueeze(0))

    gifs1 = torch.cat(tempAll, dim=0)  # torch.Size([11, 10, 1, 384, 32])
    gifs2 = torch.cat(tempAll2, dim=0)  # torch.Size([11, 10, 1, 384, 32])

    gifs = torch.cat([gifs1, gifs2], dim=3)

    # save the generated files, also the animated gifs

    mkdirs(output_dir_trvsl)
    mkdirs(out_dir)

    for j, val in enumerate(interpolation):
        # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
        I = gifs[:, j]
        save_image(
            tensor=I.cpu(),
            filename=os.path.join(out_dir, '%03d.jpg' % (j)),
            nrow=zA_dim + zS_dim,
            pad_value=1)
        # make animated gif
    grid2gif(
        out_dir, str(os.path.join(out_dir, 'traverse.gif')), delay=10
    )



def save_traverse_base(iters, data_loader, enc, dec, cuda, fixed_idxs, output_dir_trvsl, flatten_pixel=None):
    output_dir_trvsl = '../output/base_' + output_dir_trvsl
    tr_range = 2
    out_dir = os.path.join(output_dir_trvsl, str(iters) +'_'+ str(-tr_range) + '~' + str(tr_range))

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i] = data_loader.dataset.__getitem__(idx)[0]
        if flatten_pixel is not None:
            fixed_XA[i] = fixed_XA[i].view(-1, flatten_pixel)
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)

    fixed_XA = torch.stack(fixed_XA, dim=0)

    q = enc(fixed_XA, num_samples=1)
    zA_ori, zS_ori = q['styles'].dist.loc, q['digits'].value

    zA_dim = zA_ori.shape[2]
    zS_dim = zS_ori.shape[2]
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, 10))

    tempA = []  # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
    loc=-1
    for row in range(zA_dim):
        if loc != -1 and row != loc:
            continue
        zA = zA_ori.clone()

        temp = []
        for val in interpolation:
            zA[:, :, row] = val
            sampleA = dec.forward2(q['digits'].value, zA, cuda)
            if flatten_pixel is not None:
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                sampleA = torch.transpose(sampleA, 0,1)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    tempS = []

    for i in range(zS_dim):
        zS = np.zeros((1, 1, zS_dim))
        zS[0, 0, i % zS_dim] = 1.
        zS = torch.Tensor(zS)
        zS = torch.cat([zS] * len(fixed_idxs), dim=1)
        if cuda:
            zS = zS.cuda()
        sampleA = dec.forward2(zS, zA_ori, cuda)
        if flatten_pixel is not None:
            sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
            sampleA = torch.transpose(sampleA, 0, 1)
        tempS.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

    tempA.append(torch.cat(tempS, dim=0).unsqueeze(0))
    gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

    # save the generated files, also the animated gifs
    mkdirs(output_dir_trvsl)
    mkdirs(out_dir)

    for j, val in enumerate(interpolation):
        # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
        I = gifs[:, j]
        save_image(
            tensor=I.cpu(),
            filename=os.path.join(out_dir, '%03d.jpg' % (j)),
            nrow=1 + zA_dim + 1 + 1 + 1 + zS_dim,
            pad_value=1)
        # make animated gif
    grid2gif(
        out_dir, str(os.path.join(out_dir, 'traverse' + '.gif')), delay=10
    )

    # digit trv imgs

    ### orginal img ###
    digit_trv = []
    fixed_XA = fixed_XA.view(-1, 1, 28, 28)
    digit_trv.append((torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0))
    sampleA = dec.forward2(zS_ori, zA_ori, cuda)
    if flatten_pixel is not None:
        sampleA = sampleA.view(-1, 1, 28, 28)
    digit_trv.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
    WS = torch.ones(digit_trv[0].shape)
    # WS = torch.ones((sampleA.shape[0], 1, 1,1))
    if cuda:
        WS = WS.cuda()
    digit_trv.append(WS)
    digit_trv.extend(tempS)

    mkdirs(out_dir + '/digit/')
    save_image(
        tensor=(torch.cat(digit_trv, dim=0)).cpu(),
        filename=os.path.join(out_dir + '/digit/', 'digit.jpg'),
        nrow=3 + zS_dim,
        pad_value=1)
    #########

def mutual_info(data_loader, enc, cuda, flatten_pixel=None, baseline=False, plot=False):
    # fixed_idxs = [3, 2, 1, 18, 4, 15, 11, 17, 61, 99]

    num_labels = 10
    per_label_samples = 100
    per_label_cnt = {}
    for i in range(num_labels):
        per_label_cnt.update({i: 0})

    fixed_XA = []
    fixed_XB = []
    for i in range(num_labels):
        j = 0
        while per_label_cnt[i] < per_label_samples:
            img, label = data_loader.dataset.__getitem__(j)
            if label == i:
                if flatten_pixel is not None:
                    img = img.view(-1, flatten_pixel)
                if cuda:
                    img = img.cuda()
                img = img.squeeze(0)
                fixed_XA.append(img)
                fixed_XB.append(label)
                per_label_cnt[i] += 1
            j+=1

    fixed_XA = torch.stack(fixed_XA, dim=0)
    q = enc(fixed_XA, num_samples=1)
    batch_dim= 1

    if baseline:
        batch_size = q['styles'].value.shape[1]
        z_private= q['styles'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
        z_shared= q['digits'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
        q_ziCx_private = torch.exp(q['styles'].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
        q_ziCx_shared = torch.exp(q['digits'].dist.log_pmf(z_shared).transpose(1, batch_dim + 1))
        q_ziCx = torch.cat((q_ziCx_private,q_ziCx_shared), dim=2)
    else:
        # for my model
        batch_size = q['privateA'].value.shape[1]
        z_private= q['privateA'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
        z_shared= q['sharedA'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
        q_ziCx_private = torch.exp(q['privateA'].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
        q_ziCx_shared = torch.exp(q['sharedA'].dist.log_pmf(z_shared).transpose(1, batch_dim + 1))
        q_ziCx = torch.cat((q_ziCx_private,q_ziCx_shared), dim=2)


    latent_dim = q_ziCx.shape[-1]
    mi_zi_y = torch.tensor([.0] * latent_dim)
    if cuda:
        mi_zi_y = mi_zi_y.cuda()
    for k in range(num_labels):
        q_ziCxk = q_ziCx[k * per_label_samples:(k + 1) * per_label_samples, k * per_label_samples:(k + 1) * per_label_samples, :]
        marg_q_ziCxk = q_ziCxk.sum(1)
        mi_zi_y += (marg_q_ziCxk * (np.log(batch_size/num_labels) + torch.log(marg_q_ziCxk) - torch.log(q_ziCx[k * per_label_samples:(k + 1) * per_label_samples, :, :].sum(1)))).mean(0)
    mi_zi_y = mi_zi_y / batch_size
    print(mi_zi_y)

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        ax.bar(range(latent_dim), mi_zi_y.detach().cpu().numpy())
        # ax.set_xticks(range(latent_dim))
        my_xticks = []
        for i in range(latent_dim - 1):
            my_xticks.append('z' + str(i + 1))
        my_xticks.append('c')
        plt.xticks(range(latent_dim), my_xticks)
        # ax.set_title('poeA')
        plt.show()
    return mi_zi_y.detach().cpu().numpy()



def save_reconst(iters, data_loader, encA, decA, encB, decB, cuda, output_dir_trvsl, flatten_pixel=None, fixed_idxs=[3, 2, 1, 30, 4, 23, 21, 41, 84, 99]):
    EPS = 1e-9
    output_dir_trvsl = '../output/' + output_dir_trvsl
    tr_range = 2
    out_dir = os.path.join(output_dir_trvsl, str(iters) +'_'+ str(-tr_range) + '~' + str(tr_range), 'reconst')
    batch_size = len(fixed_idxs)

    images = [0] * len(fixed_idxs)
    label = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        images[i], label[i] = data_loader.dataset.__getitem__(idx)[:2]
        if flatten_pixel is not None:
            images[i] = images[i].view(-1, flatten_pixel)
        # images = fixed_imgs.view(-1, NUM_PIXELS)

        if cuda:
            images[i] = images[i].cuda()
            images[i] = images[i].squeeze(0)

    images = torch.stack(images, dim=0)
    label = torch.LongTensor(label)
    labels_onehot = torch.zeros(batch_size, 10)
    labels_onehot.scatter_(1, label.unsqueeze(1), 1)
    labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
    if cuda:
        labels_onehot = labels_onehot.cuda()

    # encode
    q = encA(images, num_samples=1)
    q = encB(labels_onehot, num_samples=1, q=q)
    ## poe ##
    prior_logit = torch.zeros_like(q['sharedA'].dist.logits)  # prior is the concrete dist. of uniform dist.
    poe_logit = q['sharedA'].dist.logits + q['sharedB'].dist.logits + prior_logit
    q.concrete(logits=poe_logit,
               temperature=0.66,
               name='poe')

    XA_infA_recon = decA.forward2(q['privateA'].dist.loc, q['sharedA'].value, cuda)
    print('sharedA')
    print(q['sharedA'].value.argmax(dim=2))
    XA_POE_recon = decA.forward2(q['privateA'].dist.loc, q['poe'].value, cuda)
    print('poe')
    print(q['poe'].value.argmax(dim=2))
    XA_sinfB_recon = decA.forward2(q['privateA'].dist.loc, q['sharedB'].value, cuda)
    print('sharedB')
    print(q['sharedB'].value.argmax(dim=2))

    if flatten_pixel is not None:
        XA_infA_recon = XA_infA_recon.view(XA_infA_recon.shape[0], -1, 28, 28)
        XA_infA_recon = torch.transpose(XA_infA_recon, 0, 1)
        XA_POE_recon = XA_POE_recon.view(XA_POE_recon.shape[0], -1, 28, 28)
        XA_POE_recon = torch.transpose(XA_POE_recon, 0, 1)
        XA_sinfB_recon = XA_sinfB_recon.view(XA_sinfB_recon.shape[0], -1, 28, 28)
        XA_sinfB_recon = torch.transpose(XA_sinfB_recon, 0, 1)


    WS = torch.ones(images.shape)
    if cuda:
        WS = WS.cuda()

    imgs = [images, XA_infA_recon, XA_POE_recon, XA_sinfB_recon, WS]
    merged = torch.cat(
        imgs, dim=0
    )

    perm = torch.arange(0, len(imgs) * batch_size).view(len(imgs), batch_size).transpose(1, 0)
    perm = perm.contiguous().view(-1)
    merged = merged[perm, :].cpu()

    # save the results as image
    fname = os.path.join(out_dir, 'reconA_%s.jpg' % iters)
    mkdirs(out_dir)
    save_image(
        tensor=merged, filename=fname, nrow=len(imgs) * int(np.sqrt(batch_size)),
        pad_value=1
    )


def make_one_hot(alpha, cuda):
    _, max_alpha = torch.max(alpha, dim=1)
    one_hot_samples = torch.zeros(alpha.size())
    one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
    if cuda:
        one_hot_samples = one_hot_samples.cuda()
    return one_hot_samples


def save_reconst_awa(iters, data_loader, enc, dec, cuda, output_dir_trvsl, n_shared,
                     fixed_idxs=[3, 2, 1, 30, 4, 23, 21, 41, 84, 99]):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    tr_range = 2
    out_dir = os.path.join(output_dir_trvsl, str(iters) + '_' + str(-tr_range) + '~' + str(tr_range), 'reconst')
    batch_size = len(fixed_idxs)

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zS_ori = []
    for i in range(n_shared):
        zShared = q['sharedA' + str(i)].value
        zShared = make_one_hot(zShared.squeeze(0), cuda)
        zS_ori.append(zShared.unsqueeze(0))

    latents = [q['privateA'].dist.loc]
    latents.extend(zS_ori)
    XA_infA_recon = dec.forward2(latents, cuda)

    WS = torch.ones(fixed_XA.shape)
    if cuda:
        WS = WS.cuda()

    imgs = [fixed_XA, XA_infA_recon, WS]
    merged = torch.cat(
        imgs, dim=0
    )

    perm = torch.arange(0, len(imgs) * batch_size).view(len(imgs), batch_size).transpose(1, 0)
    perm = perm.contiguous().view(-1)
    merged = merged[perm, :].cpu()

    # save the results as image
    fname = os.path.join(out_dir, 'reconA_%s.jpg' % iters)
    mkdirs(out_dir)
    save_image(
        tensor=merged, filename=fname, nrow=len(imgs) * int(np.sqrt(batch_size)),
        pad_value=1
    )


def save_traverse_awa(iters, data_loader, enc, dec, cuda, output_dir_trvsl, n_shared,
                      fixed_idxs=[0]):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    tr_range = 2
    out_dir = os.path.join(output_dir_trvsl, str(iters) + '_' + str(-tr_range) + '~' + str(tr_range), str(fixed_idxs),
                           'private')

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    zS_ori = []
    for i in range(n_shared):
        zS_ori.append(q['sharedA' + str(i)].value)

    latents = [zA_ori]
    latents.extend(zS_ori)
    recon_img = dec.forward2(latents, cuda)

    zA_dim = zA_ori.shape[2]
    zS_dim = zS_ori[0].shape[2]

    n_interp = 5
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, n_interp))

    tempA = []
    loc = -1
    for row in range(zA_dim):
        if loc != -1 and row != loc:
            continue
        zA = zA_ori.clone()

        temp = []
        for val in interpolation:
            zA[:, :, row] = val
            latents = [zA]
            latents.extend(zS_ori)
            sampleA = dec.forward2(latents, cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    temp = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * n_interp
    tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
    temp = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0)] * n_interp
    tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
    gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

    # save the generated files, also the animated gifs

    mkdirs(output_dir_trvsl)
    mkdirs(out_dir)

    for j, val in enumerate(interpolation):
        # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
        save_image(
            tensor=gifs[:, j].cpu(),
            filename=os.path.join(out_dir, '%03d.jpg' % (j)),
            nrow=1 + zA_dim + 1,
            pad_value=1)
        # make animated gif

    grid2gif(
        out_dir, str(os.path.join(out_dir, 'traverse_private' + '.gif')), delay=10
    )
    del tempA
    del temp
    del gifs
    del zA
    del latents

    tempS = []
    for row in range(n_shared):
        if loc != -1 and row != loc:
            continue
        zS = zS_ori.copy()
        temp = []
        for i in range(zS_dim):
            one_hot = torch.zeros_like(zS[row])
            one_hot[:, :, i % zS_dim] = 1
            zS[row] = one_hot
            if cuda:
                zS[row] = zS[row].cuda()
            latents = [zA_ori]
            latents.extend(zS)
            sampleA = dec.forward2(latents, cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempS.append(torch.cat(temp, dim=0).unsqueeze(0))

    temp = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * 2
    tempS.append(torch.cat(temp, dim=0).unsqueeze(0))
    temp = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0)] * 2
    tempS.append(torch.cat(temp, dim=0).unsqueeze(0))
    gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])

    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, zS_dim))

    output_dir_trvsl = '../output/' + output_dir_trvsl
    tr_range = 2
    out_dir = os.path.join(output_dir_trvsl, str(iters) + '_' + str(-tr_range) + '~' + str(tr_range), str(fixed_idxs),
                           'shared')
    mkdirs(output_dir_trvsl)
    mkdirs(out_dir)

    for j, val in enumerate(interpolation):
        save_image(
            tensor=gifs_shared[:, j].cpu(),
            filename=os.path.join(out_dir, '%03d.jpg' % (j)),
            nrow=1 + n_shared + 1,
            pad_value=1)
        # make animated gif
    grid2gif(
        out_dir, str(os.path.join(out_dir, 'traverse_shared' + '.gif')), delay=10
    )


def save_traverse_cub(iters, data_loader, enc, dec, cuda, output_dir_trvsl, attr_dim,
                      fixed_idxs=[0], private=True, label=False):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    mkdirs(output_dir_trvsl)

    tr_range = 2

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    zS_attr_ori = []
    zS_label_ori = q['sharedA_label'].value
    for i in range(len(attr_dim)):
        zS_attr_ori.append(q['sharedA_attr' + str(i)].value)

    latents = [zA_ori]
    latents.extend(zS_attr_ori)
    latents.append(zS_label_ori)
    recon_img = dec.forward2(latents, cuda)

    zA_dim = zA_ori.shape[2]

    n_interp = 5
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, n_interp))
    loc = -1

    if private:
        tempA = []
        for row in range(zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolation:
                zA[:, :, row] = val
                latents = [zA]
                latents.extend(zS_attr_ori)
                latents.append(zS_label_ori)
                sampleA = dec.forward2(latents, cuda)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

            tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

        temp = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        temp = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        # save the generated files, also the animated gifs

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'private_range' + str(tr_range))
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            save_image(
                tensor=gifs[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + zA_dim + 1,
                pad_value=1)
            # make animated gif

        grid2gif(
            out_dir, str(os.path.join(out_dir, 'traverse_private' + '.gif')), delay=10
        )
        del tempA
        del temp
        del gifs
        del zA
        del latents

    # for shared attr
    for row in range(len(attr_dim)):
        tempS = []
        if loc != -1 and row != loc:
            continue
        zS = zS_attr_ori.copy()
        this_attr_dim = attr_dim[row]

        # add original, reconstructed img
        gt_img = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * this_attr_dim
        reconst_img = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(
            0)] * this_attr_dim
        tempS.append(torch.cat(gt_img, dim=0).unsqueeze(0))
        tempS.append(torch.cat(reconst_img, dim=0).unsqueeze(0))

        temp = []
        for i in range(this_attr_dim):
            one_hot = torch.zeros_like(zS[row])
            one_hot[:, :, i % this_attr_dim] = 1
            zS[row] = one_hot
            if cuda:
                zS[row] = zS[row].cuda()
            latents = [zA_ori]
            latents.extend(zS)
            latents.append(zS_label_ori)
            sampleA = dec.forward2(latents, cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempS.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        interpolation = torch.tensor(np.linspace(-tr_range, tr_range, this_attr_dim))

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'attr' + str(row) + '_dim' + str(this_attr_dim))
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            save_image(
                tensor=gifs_shared[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + 1 + 1,
                pad_value=1)
            # make animated gif
        file_name = str(iters) + 'iter_attr' + str(row) + '_dim' + str(this_attr_dim)
        grid2gif(
            out_dir, str(os.path.join(out_dir, file_name + '.gif')), delay=10
        )



    ### for label
    if label:
        tempS = []
        zS = zS_label_ori.clone()
        class_dim = 200

        temp = []
        for i in range(class_dim):
            one_hot = torch.zeros_like(zS)
            one_hot[:, :, i % class_dim] = 1
            zS = one_hot
            if cuda:
                zS = zS.cuda()
            latents = [zA_ori]
            latents.extend(zS_attr_ori)
            latents.append(zS)
            sampleA = dec.forward2(latents, cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempS.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        interpolation = torch.tensor(np.linspace(-tr_range, tr_range, class_dim))

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'label_dim200')
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            save_image(
                tensor=gifs_shared[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + 1 + 1,
                pad_value=1)
            # make animated gif
        file_name = str(iters) + 'iter_label_dim200'
        grid2gif(
            out_dir, str(os.path.join(out_dir, file_name + '.gif')), delay=10
        )


def save_recon_cub(iters, data_loader, enc, dec, encB, cuda, output_dir_trvsl, attr_dim,
                   fixed_idxs=[0]):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    mkdirs(output_dir_trvsl)

    tr_range = 2

    imgs = [0] * len(fixed_idxs)
    attr = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        imgs[i], attr[i] = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            imgs[i] = imgs[i].cuda()
            imgs[i] = imgs[i].squeeze(0)
    imgs = torch.stack(imgs, dim=0)
    attributes = []
    for i in range(imgs.shape[0]):
        concat_all_attr = []
        for j in range(len(attr_dim)):
            concat_all_attr.append(torch.Tensor(attr[i][j]))
        attributes.append(torch.cat(concat_all_attr, dim=0))
    attributes = torch.stack(attributes).float()
    if cuda:
        attributes = attributes.cuda()

    # do traversal and collect generated images

    q = enc(imgs, num_samples=1)
    q = encB(attributes, q=q, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    sharedA_attr = []
    sharedB_attr = []
    for i in range(len(attr_dim)):
        sharedA_attr.append(q['sharedA_attr' + str(i)].value)
        sharedB_attr.append(q['sharedB_attr' + str(i)].value)

    recon_img = dec.forward2([zA_ori] + sharedA_attr, cuda)
    recon_img_cr = dec.forward2([zA_ori] + sharedB_attr, cuda)

    temp = []
    temp.append((torch.cat([imgs[i] for i in range(imgs.shape[0])], dim=1)).unsqueeze(0))
    temp.append((torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0))
    temp.append((torch.cat([recon_img_cr[i] for i in range(recon_img_cr.shape[0])], dim=1)).unsqueeze(0))

    fin = torch.cat(temp, dim=0)

    # save the generated files, also the animated gifs

    out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs))
    mkdirs(out_dir)
    save_image(
        tensor=fin[:, 0].cpu(),
        filename=os.path.join(out_dir, 'rec.jpg'),
        nrow=1 + 1 + 1,
        pad_value=1)


def save_traverse_cub_ia(iters, data_loader, enc, dec, cuda, output_dir_trvsl, attr_dim,
                         fixed_idxs=[0], private=True):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    mkdirs(output_dir_trvsl)

    tr_range = 2

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    zS_attr_ori = []
    for i in range(len(attr_dim)):
        zS_attr_ori.append(q['sharedA_attr' + str(i)].value)

    latents = [zA_ori]
    latents.extend(zS_attr_ori)
    recon_img = dec.forward2(latents, cuda)

    zA_dim = zA_ori.shape[2]

    n_interp = 5
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, n_interp))
    loc = -1

    if private:
        tempA = []
        for row in range(zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolation:
                zA[:, :, row] = val
                latents = [zA]
                latents.extend(zS_attr_ori)
                sampleA = dec.forward2(latents, cuda)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

            tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

        temp = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        temp = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        # save the generated files, also the animated gifs

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'private_range' + str(tr_range))
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            save_image(
                tensor=gifs[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + zA_dim + 1,
                pad_value=1)
            # make animated gif

        grid2gif(
            out_dir, str(os.path.join(out_dir, 'traverse_private' + '.gif')), delay=10
        )
        del tempA
        del temp
        del gifs
        del zA
        del latents

    # for shared attr
    for row in range(len(attr_dim)):
        tempS = []
        if loc != -1 and row != loc:
            continue
        zS = zS_attr_ori.copy()
        this_attr_dim = attr_dim[row]

        # add original, reconstructed img
        gt_img = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * this_attr_dim
        reconst_img = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(
            0)] * this_attr_dim
        tempS.append(torch.cat(gt_img, dim=0).unsqueeze(0))
        tempS.append(torch.cat(reconst_img, dim=0).unsqueeze(0))

        temp = []
        for i in range(this_attr_dim):
            one_hot = torch.zeros_like(zS[row])
            one_hot[:, :, i % this_attr_dim] = 1
            zS[row] = one_hot
            if cuda:
                zS[row] = zS[row].cuda()
            latents = [zA_ori]
            latents.extend(zS)
            sampleA = dec.forward2(latents, cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempS.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        interpolation = torch.tensor(np.linspace(-tr_range, tr_range, this_attr_dim))

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'attr' + str(row) + '_dim' + str(this_attr_dim))
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            save_image(
                tensor=gifs_shared[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + 1 + 1,
                pad_value=1)
            # make animated gif
        file_name = str(iters) + 'iter_attr' + str(row) + '_dim' + str(this_attr_dim)
        grid2gif(
            out_dir, str(os.path.join(out_dir, file_name + '.gif')), delay=10
        )


def save_traverse_cub_ia2(iters, data_loader, enc, dec, cuda, output_dir_trvsl, attr_dim,
                          fixed_idxs=[0], private=True):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    mkdirs(output_dir_trvsl)

    tr_range = 2

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    zS_attr_ori = []

    for i in range(sum(attr_dim)):
        zS_attr_ori.append(q['sharedA_attr' + str(i)].value)

    latents = [zA_ori]
    latents.extend(zS_attr_ori)
    recon_img = dec.forward2(latents, cuda)

    zA_dim = zA_ori.shape[2]

    n_interp = 5
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, n_interp))
    loc = -1

    if private:
        tempA = []
        for row in range(zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolation:
                zA[:, :, row] = val
                latents = [zA]
                latents.extend(zS_attr_ori)
                sampleA = dec.forward2(latents, cuda)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

            tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

        temp = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        temp = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        # save the generated files, also the animated gifs

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'private_range' + str(tr_range))
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            save_image(
                tensor=gifs[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + zA_dim + 1,
                pad_value=1)
            # make animated gif

        grid2gif(
            out_dir, str(os.path.join(out_dir, 'traverse_private' + '.gif')), delay=10
        )
        del tempA
        del temp
        del gifs
        del zA
        del latents

    # for shared attr
    row = 0
    for i in range(len(attr_dim)):
        for j in range(attr_dim[i]):
            tempS = []
            if loc != -1 and row != loc:
                continue
            zS = zS_attr_ori.copy()
            this_attr_dim = 2

            # add original, reconstructed img
            gt_img = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * this_attr_dim
            reconst_img = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(
                0)] * this_attr_dim
            tempS.append(torch.cat(gt_img, dim=0).unsqueeze(0))
            tempS.append(torch.cat(reconst_img, dim=0).unsqueeze(0))

            temp = []
            for d in range(this_attr_dim):
                one_hot = torch.zeros_like(zS[row])
                one_hot[:, :, d % this_attr_dim] = 1
                zS[row] = one_hot
                if cuda:
                    zS[row] = zS[row].cuda()
                latents = [zA_ori]
                latents.extend(zS)
                sampleA = dec.forward2(latents, cuda)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
            tempS.append(torch.cat(temp, dim=0).unsqueeze(0))
            gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])

            interpolation = torch.tensor(np.linspace(-tr_range, tr_range, this_attr_dim))

            out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                                   'attr' + str(i) + '_class' + str(j))
            mkdirs(out_dir)

            for j, val in enumerate(interpolation):
                save_image(
                    tensor=gifs_shared[:, j].cpu(),
                    filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                    nrow=1 + 1 + 1,
                    pad_value=1)
                # make animated gif
            file_name = str(iters) + 'iter_attr' + str(i) + '_class' + str(j)
            grid2gif(
                out_dir, str(os.path.join(out_dir, file_name + '.gif')), delay=10
            )
            row += 1


def save_traverse_celeba_cont(iters, data_loader, enc, dec, cuda, output_dir_trvsl,
                              fixed_idxs=[0], private=True, min=-2, max=2):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    mkdirs(output_dir_trvsl)


    tr_range = 3

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    zS_ori = q['sharedA'].dist.loc

    latents = [zA_ori, zS_ori]
    recon_img = dec.forward2(latents, cuda)

    zA_dim = zA_ori.shape[2]
    zS_dim = zS_ori.shape[2]

    n_interp = 10
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, n_interp))

    loc = -1

    if private:
        tempA = []
        for row in range(zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolation:
                zA[:, :, row] = val
                sampleA = dec.forward2([zA, zS_ori], cuda)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        temp = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        temp = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        # save the generated files, also the animated gifs

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'private_range' + str(tr_range))
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            save_image(
                tensor=gifs[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + zA_dim + 1,
                pad_value=1)
            # make animated gif

        grid2gif(
            out_dir, str(os.path.join(out_dir, 'traverse_private' + '.gif')), delay=10
        )
        del tempA
        del temp
        del gifs
        del zA
        del latents

    #### shared1
    # for shared attr
    tempS = []
    interpolation = torch.tensor(np.linspace(min, max, n_interp))
    interpolation = torch.transpose(interpolation, 1, 0)


    # add original, reconstructed img
    gt_img = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * n_interp
    reconst_img = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(
        0)] * n_interp
    tempS.append(torch.cat(gt_img, dim=0).unsqueeze(0))
    tempS.append(torch.cat(reconst_img, dim=0).unsqueeze(0))
    for row in range(int(zS_dim / 2)):
        if loc != -1 and row != loc:
            continue
        zS = zS_ori.clone()
        temp = []
        for val in interpolation[row]:
            zS[:, :, row] = val
            sampleA = dec.forward2([zA_ori, zS], cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempS.append(torch.cat(temp, dim=0).unsqueeze(0))

    gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])
    out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs))
    part1_dir = os.path.join(out_dir, 'part1')
    mkdirs(out_dir)
    mkdirs(part1_dir)

    for j in range(n_interp):
        save_image(
            tensor=gifs_shared[:, j].cpu(),
            filename=os.path.join(part1_dir, '%03d.jpg' % (j)),
            nrow=2 + int(zS_dim / 2),
            pad_value=1)
        # make animated gif
    grid2gif(
        part1_dir, str(os.path.join(out_dir, 'part1' + '.gif')), delay=10, duration=0.12
    )

    #### shared2
    # for shared attr
    tempS = []
    for row in range(int(zS_dim / 2), zS_dim):
        if loc != -1 and row != loc:
            continue
        zS = zS_ori.clone()
        temp = []
        for val in interpolation[row]:
            zS[:, :, row] = val
            sampleA = dec.forward2([zA_ori, zS], cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempS.append(torch.cat(temp, dim=0).unsqueeze(0))

    gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])

    part2_dir = os.path.join(out_dir, 'part2')
    mkdirs(part2_dir)

    for j in range(n_interp):
        save_image(
            tensor=gifs_shared[:, j].cpu(),
            filename=os.path.join(part2_dir, '%03d.jpg' % (j)),
            nrow=int(zS_dim / 2),
            pad_value=1)
        # make animated gif
    grid2gif(
        part2_dir, str(os.path.join(out_dir, 'part2' + '.gif')), delay=10, duration=0.12
    )
    np.savetxt(os.path.join(out_dir, 'min.txt'), min)
    np.savetxt(os.path.join(out_dir, 'max.txt'), max)


def save_traverse_celeba(iters, data_loader, enc, dec, zS_dim, cuda, output_dir_trvsl,
                         fixed_idxs=[0], private=True):
    output_dir_trvsl = '../output/' + output_dir_trvsl
    mkdirs(output_dir_trvsl)

    tr_range = 2

    fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    # do traversal and collect generated images

    q = enc(fixed_XA, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    zS_ori = []
    for i in range(zS_dim):
        zS_ori.append(q['sharedA' + str(i)].value)

    latents = [zA_ori]
    latents.extend(zS_ori)
    recon_img = dec.forward2(latents, cuda)

    zA_dim = zA_ori.shape[2]

    n_interp = 5
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, n_interp))
    loc = -1

    if private:
        tempA = []
        for row in range(zA_dim):
            if loc != -1 and row != loc:
                continue
            zA = zA_ori.clone()

            temp = []
            for val in interpolation:
                zA[:, :, row] = val
                sampleA = dec.forward2([zA, zS_ori], cuda)
                temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        temp = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        temp = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(0)] * n_interp
        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
        gifs = torch.cat(tempA, dim=0)  # torch.Size([11, 10, 1, 384, 32])

        # save the generated files, also the animated gifs

        out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                               'private_range' + str(tr_range))
        mkdirs(out_dir)

        for j, val in enumerate(interpolation):
            # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
            save_image(
                tensor=gifs[:, j].cpu(),
                filename=os.path.join(out_dir, '%03d.jpg' % (j)),
                nrow=1 + zA_dim + 1,
                pad_value=1)
            # make animated gif

        grid2gif(
            out_dir, str(os.path.join(out_dir, 'traverse_private' + '.gif')), delay=10
        )
        del tempA
        del temp
        del gifs
        del zA
        del latents

    # for shared attr
    tempS = []
    this_attr_dim = 2
    # add original, reconstructed img
    gt_img = [(torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=1)).unsqueeze(0)] * this_attr_dim
    reconst_img = [(torch.cat([recon_img[i] for i in range(recon_img.shape[0])], dim=1)).unsqueeze(
        0)] * this_attr_dim
    tempS.append(torch.cat(gt_img, dim=0).unsqueeze(0))
    tempS.append(torch.cat(reconst_img, dim=0).unsqueeze(0))
    for row in range(zS_dim):
        if loc != -1 and row != loc:
            continue
        zS = zS_ori.copy()
        temp = []
        for d in range(this_attr_dim):
            one_hot = torch.zeros_like(zS[row])
            one_hot[:, :, d % this_attr_dim] = 1
            zS[row] = one_hot
            if cuda:
                zS[row] = zS[row].cuda()
            latents = [zA_ori]
            latents.extend(zS)
            sampleA = dec.forward2(latents, cuda)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempS.append(torch.cat(temp, dim=0).unsqueeze(0))

    gifs_shared = torch.cat(tempS, dim=0)  # torch.Size([11, 10, 1, 384, 32])
    out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs),
                           'shared_' + str(zS_dim))
    mkdirs(out_dir)
    interpolation = torch.tensor(np.linspace(-tr_range, tr_range, this_attr_dim))
    for j, val in enumerate(interpolation):
        save_image(
            tensor=gifs_shared[:, j].cpu(),
            filename=os.path.join(out_dir, '%03d.jpg' % (j)),
            nrow=2 + zS_dim,
            pad_value=1)
        # make animated gif
    grid2gif(
        out_dir, str(os.path.join(out_dir, 'traverse_shared' + '.gif')), delay=10, duration=0.5
    )


def load_classifier(cuda):
    sys.path.append('../')
    from celeba_classifier.model import EncoderA
    model_name = '../weights/celeba_clf/clf.rar'
    encA = EncoderA(0, n_attr=18)
    if cuda:
        encA.load_state_dict(torch.load(model_name))
    else:
        encA.load_state_dict(torch.load(model_name,
                                        map_location=torch.device('cpu')))
    return encA


def save_cross_celeba(iters, data_loader, encA, decA, encB, gt_attrs, n_samples, zS_dim, cuda, output_dir,
                      acc_check=False):
    output_dir = '../output/' + output_dir + '/cross' + str(iters)
    mkdirs(output_dir)

    # # using training stat
    # # img private
    # n_batch = 10
    # fixed_idxs = random.sample(range(len(data_loader.dataset)), 100 * n_batch)
    # fixed_XA = [0] * 100 * n_batch
    # for i, idx in enumerate(fixed_idxs):
    #     fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
    #     if cuda:
    #         fixed_XA[i] = fixed_XA[i].cuda()
    #     fixed_XA[i] = fixed_XA[i].squeeze(0)
    # fixed_XA = torch.stack(fixed_XA, dim=0)
    #
    # zA_mean = 0
    # zA_std = 0
    # # zS_ori_sum = np.zeros(zS_dim)
    # for idx in range(n_batch):
    #     q = encA(fixed_XA[100*idx:100*(idx+1)], num_samples=1)
    #     zA_mean += q['privateA'].dist.loc
    #     zA_std += q['privateA'].dist.scale
    #     # zS_ori = []
    #     # for i in range(zS_dim):
    #     #     zS_ori.append(q['sharedA' + str(i)].value)
    #     # zS_ori_sum += np.array(zS_ori)
    # zA_mean = zA_mean.mean(dim=1)
    # zA_std = zA_std.mean(dim=1)
    #
    # q.normal(loc=zA_mean,
    #              scale=zA_std,
    #              name='sample')
    # zA = []
    # for _ in range(n_samples):
    #     zA.append(q['sample'].dist.sample().unsqueeze(1))
    # zA = torch.cat(zA, dim=1)
    #############################################


    ########################
    # using test
    # img private
    n_batch = 10
    torch.manual_seed(0)
    random.seed(0)
    fixed_idxs = random.sample(range(len(data_loader.dataset)), n_samples)
    fixed_XA = [0] * n_samples
    attributes = [0] * n_samples
    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], attributes[i] = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)
    attributes = torch.stack(attributes, dim=0)

    save_image(fixed_XA.view(n_samples, 3, 64, 64),
               str(os.path.join(output_dir, 'gt_image.png')), nrow=int(np.sqrt(n_samples)))

    q = encA(fixed_XA, num_samples=1)
    zA = q['privateA'].dist.sample()
    ########################

    # attr shared

    gt_attrs = ['recon'] + gt_attrs
    for gt_attr in gt_attrs:
        if 'recon' in gt_attr:
            # img shared
            zS = []
            for i in range(zS_dim):
                zS.append(q['sharedA' + str(i)].value)
        else:
            attrs = torch.zeros(zS_dim)
            if 'off' not in gt_attr:
                attr_ix = ATTR_IX_TO_KEEP.index(ATTR_TO_IX_DICT[gt_attr])
                attrs[attr_ix] = 1
            zS = []
            if cuda:
                attrs = attrs.cuda()
            attrs = attrs.repeat((n_samples, 1))
            q = encB(attrs, num_samples=1)
            for i in range(zS_dim):
                zS.append(q['sharedB' + str(i)].value)

        latents = [zA] + zS
        recon_img = decA.forward2(latents, cuda)
        if acc_check:
            if gt_attr not in ('recon', 'off'):
                clf = load_classifier(cuda)
                pred_attr = clf(recon_img, num_samples=1)

                if cuda:
                    pred_attr = pred_attr.cpu()

                pred = pred_attr.detach().numpy()
                pred = np.round(np.exp(pred))
                target = np.ones_like(pred)
                attr_idx = ATTR_IX_TO_KEEP.index(ATTR_TO_IX_DICT[gt_attr])
                acc = (pred[:, attr_idx] == target[:, attr_idx]).mean()
                f1 = f1_score(target[:, attr_idx], pred[:, attr_idx], average="binary")
                print('--------' + gt_attr + '--------')
                print('acc:', acc)
                print('f1:', f1)

        else:
            save_image(recon_img.view(n_samples, 3, 64, 64),
                       str(os.path.join(output_dir, gt_attr + '_image_iter.png')), nrow=int(np.sqrt(n_samples)))



def cross_acc_celeba(iters, data_loader, encA, decA, encB, n_samples, zS_dim, cuda, output_dir):
    output_dir = '../output/' + output_dir + '/cross' + str(iters)
    mkdirs(output_dir)

    # # using training stat
    # # img private
    # n_batch = 10
    # fixed_idxs = random.sample(range(len(data_loader.dataset)), 100 * n_batch)
    # fixed_XA = [0] * 100 * n_batch
    # for i, idx in enumerate(fixed_idxs):
    #     fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
    #     if cuda:
    #         fixed_XA[i] = fixed_XA[i].cuda()
    #     fixed_XA[i] = fixed_XA[i].squeeze(0)
    # fixed_XA = torch.stack(fixed_XA, dim=0)
    #
    # zA_mean = 0
    # zA_std = 0
    # # zS_ori_sum = np.zeros(zS_dim)
    # for idx in range(n_batch):
    #     q = encA(fixed_XA[100*idx:100*(idx+1)], num_samples=1)
    #     zA_mean += q['privateA'].dist.loc
    #     zA_std += q['privateA'].dist.scale
    #     # zS_ori = []
    #     # for i in range(zS_dim):
    #     #     zS_ori.append(q['sharedA' + str(i)].value)
    #     # zS_ori_sum += np.array(zS_ori)
    # zA_mean = zA_mean.mean(dim=1)
    # zA_std = zA_std.mean(dim=1)
    #
    # q.normal(loc=zA_mean,
    #              scale=zA_std,
    #              name='sample')
    # zA = []
    # for _ in range(n_samples):
    #     zA.append(q['sample'].dist.sample().unsqueeze(1))
    # zA = torch.cat(zA, dim=1)
    #############################################


    ########################
    # using test
    # img private
    n_batch = 10
    torch.manual_seed(0)
    random.seed(0)
    fixed_idxs = random.sample(range(len(data_loader.dataset)), n_samples)
    fixed_XA = [0] * n_samples
    attributes = [0] * n_samples
    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], attributes[i] = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    q = encA(fixed_XA, num_samples=1)
    zA = q['privateA'].dist.sample()
    clf = load_classifier(cuda)
    ########################

    # attr shared
    all_acc = []
    all_f1 = []

    for i in range(18):
        gt_attr = IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]]
        attrs = torch.zeros(18)
        attrs[i] = 1
        zS = []
        if cuda:
            attrs = attrs.cuda()
        attrs = attrs.repeat((n_samples, 1))
        q = encB(attrs, num_samples=1)
        for i in range(zS_dim):
            zS.append(q['sharedB' + str(i)].value)

        latents = [zA] + zS
        recon_img = decA.forward2(latents, cuda)

        pred_attr = clf(recon_img, num_samples=1)

        if cuda:
            pred_attr = pred_attr.cpu()

        pred = pred_attr.detach().numpy()
        pred = np.round(np.exp(pred))
        target = np.ones_like(pred)
        acc = (pred[:, i] == target[:, i]).mean()
        f1 = f1_score(target[:, i], pred[:, i], average="binary")
        all_acc.append(acc)
        all_f1.append(f1)
    for i in range(18):
        print(IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]])
    print(all_acc)
    print(all_f1)



def save_cross_celeba_cont(iters, data_loader, encA, decA, encB, gt_attrs, n_samples, zS_dim, cuda, output_dir):
    output_dir = '../output/' + output_dir + '/cross' + str(iters)
    mkdirs(output_dir)

    #############################################
    # # using training stat
    # # img private
    # n_batch = 10
    # fixed_idxs = random.sample(range(len(data_loader.dataset)), 100 * n_batch)
    # fixed_XA = [0] * 100 * n_batch
    # for i, idx in enumerate(fixed_idxs):
    #     fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
    #     if cuda:
    #         fixed_XA[i] = fixed_XA[i].cuda()
    #     fixed_XA[i] = fixed_XA[i].squeeze(0)
    # fixed_XA = torch.stack(fixed_XA, dim=0)
    #
    # zA_mean = 0
    # zA_std = 0
    # # zS_ori_sum = np.zeros(zS_dim)
    # for idx in range(n_batch):
    #     q = encA(fixed_XA[100*idx:100*(idx+1)], num_samples=1)
    #     zA_mean += q['privateA'].dist.loc
    #     zA_std += q['privateA'].dist.scale
    #     # zS_ori = []
    #     # for i in range(zS_dim):
    #     #     zS_ori.append(q['sharedA' + str(i)].value)
    #     # zS_ori_sum += np.array(zS_ori)
    # zA_mean = zA_mean.mean(dim=1)
    # zA_std = zA_std.mean(dim=1)
    #
    # q.normal(loc=zA_mean,
    #              scale=zA_std,
    #              name='sample')
    # zA = []
    # for _ in range(n_samples):
    #     zA.append(q['sample'].dist.sample().unsqueeze(1))
    # zA = torch.cat(zA, dim=1)
    #############################################


    ########################
    # using test
    # img private
    n_batch = 10
    torch.manual_seed(0)
    random.seed(0)
    fixed_idxs = random.sample(range(len(data_loader.dataset)), n_samples)
    fixed_XA = [0] * n_samples
    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)
    save_image(fixed_XA.view(n_samples, 3, 64, 64),
               str(os.path.join(output_dir, 'gt_image.png')))

    q = encA(fixed_XA, num_samples=1)
    zA = q['privateA'].dist.sample()
    ########################

    gt_attrs = ['recon'] + gt_attrs
    for gt_attr in gt_attrs:
        if 'recon' in gt_attr:
            # img shared
            zS = q['sharedA'].dist.loc
        else:
            # attr shared
            attrs = torch.zeros(zS_dim)
            if 'off' not in gt_attr:
                attr_ix = ATTR_IX_TO_KEEP.index(ATTR_TO_IX_DICT[gt_attr])
                attrs[attr_ix] = 1
            if cuda:
                attrs = attrs.cuda()
            attrs = attrs.repeat((n_samples, 1))
            q = encB(attrs, num_samples=1)
            zS = q['sharedB'].dist.sample()

        latents = [zA, zS]
        recon_img = decA.forward2(latents, cuda)
        save_image(recon_img.view(n_samples, 3, 64, 64),
                   str(os.path.join(output_dir, gt_attr + '_image.png')))


def save_cross_celeba_mvae(iters, decA, encB, gt_attrs, n_samples, n_attr, cuda, output_dir):
    output_dir = '../output/' + output_dir + '/cross' + str(iters)
    mkdirs(output_dir)

    for gt_attr in gt_attrs:
        # attr shared
        attrs = torch.zeros(n_attr)
        if 'off' not in gt_attr:
            attr_ix = ATTR_IX_TO_KEEP.index(ATTR_TO_IX_DICT[gt_attr])
            attrs[attr_ix] = 1
        if cuda:
            attrs = attrs.cuda()
        attrs = attrs.unsqueeze(dim=0)
        q = encB(attrs, cuda)

        muB, stdB = probtorch.util.apply_poe(cuda, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
        q['sharedB'].dist.loc = muB
        q['sharedB'].dist.scale = stdB

        torch.manual_seed(0)
        zS = []
        for _ in range(n_samples):
            zS.append(q['sharedB'].dist.sample().squeeze(0))
        zS = torch.cat(zS, dim=0)

        recon_img = decA.forward2(zS, cuda)
        save_image(recon_img.view(n_samples, 3, 64, 64),
                   str(os.path.join(output_dir, gt_attr + '_image.png')))


def save_cross_mnist_mvae(iters, decA, encB, n_samples, cuda, output_dir):
    output_dir = '../output/' + output_dir + '/cross' + str(iters)
    mkdirs(output_dir)

    for i in range(10):
        # label = torch.zeros(n_samples, dtype=torch.int64) + i
        label = torch.tensor(i)
        if cuda:
            label = label.cuda()
        # label = label.unsqueeze(0)
        q = encB(label, cuda, num_samples=1)

        muB, stdB = probtorch.util.apply_poe(cuda, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
        q['sharedB'].dist.loc = muB
        q['sharedB'].dist.scale = stdB

        torch.manual_seed(0)
        zS = []
        for _ in range(n_samples):
            zS.append(q['sharedB'].dist.sample().unsqueeze(0))
        zS = torch.cat(zS, dim=1)

        recon_img = decA.forward2(zS, cuda)
        # recon_img = decA.forward2(muB, cuda)
        save_image(recon_img.view(n_samples, 1, 28, 28),
                   str(os.path.join(output_dir, str(i) + '_image_iter.png')))


def save_cross_mnist_base(iters, decA, encB, n_samples, cuda, output_dir):
    output_dir = '../output/' + output_dir + '/cross' + str(iters)
    mkdirs(output_dir)

    for i in range(10):
        # label = torch.zeros(n_samples, dtype=torch.int64) + i
        label = torch.tensor(i)
        if cuda:
            label = label.cuda()
        # label = label.unsqueeze(0)
        q = encB(label, cuda, num_samples=1)

        muB, stdB = probtorch.util.apply_poe(cuda, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
        q['sharedB'].dist.loc = muB
        q['sharedB'].dist.scale = stdB

        torch.manual_seed(0)
        zS = []
        for _ in range(n_samples):
            zS.append(q['sharedB'].dist.sample().unsqueeze(0))
        zS = torch.cat(zS, dim=1)

        recon_img = decA.forward2(zS, cuda)
        # recon_img = decA.forward2(muB, cuda)
        save_image(recon_img.view(n_samples, 1, 28, 28),
                   str(os.path.join(output_dir, str(i) + '_image_iter.png')))


def save_cross_mnist(iters, data_loader, encA, decA, encB, n_samples, zS_dim, cuda, output_dir, flatten_pixel=None):
    output_dir = '../output/' + output_dir + '/cross' + str(iters)
    mkdirs(output_dir)

    torch.manual_seed(0)
    random.seed(0)
    fixed_idxs = random.sample(range(len(data_loader.dataset)), n_samples)
    fixed_XA = [0] * n_samples

    # fixed_XA = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if flatten_pixel is not None:
            fixed_XA[i] = fixed_XA[i].view(-1, flatten_pixel)
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)

    fixed_XA = torch.stack(fixed_XA, dim=0)

    if flatten_pixel:
        save_image(fixed_XA.view(n_samples, 1, 28, 28),
                   str(os.path.join(output_dir, 'gt_image.png')))
    else:
        save_image(fixed_XA,
                   str(os.path.join(output_dir, 'gt_image.png')))
    q = encA(fixed_XA, num_samples=1)
    zA = q['privateA'].dist.loc
    ########################

    # label shared
    recon_img = decA.forward2(zA, q['sharedA'].value, cuda)
    if flatten_pixel:
        save_image(recon_img.view(n_samples, 1, 28, 28),
                   str(os.path.join(output_dir, 'recon_image.png')))
    else:
        save_image(recon_img,
                   str(os.path.join(output_dir, 'recon_image.png')))

    for i in range(10):
        label = torch.zeros(zS_dim)
        label[i] = 1

        if cuda:
            label = label.cuda()
        label = label.repeat((n_samples, 1))
        q = encB(label, num_samples=1)
        recon_img = decA.forward2(zA, q['sharedB'].value, cuda)
        if flatten_pixel:
            save_image(recon_img.view(n_samples, 1, 28, 28),
                       str(os.path.join(output_dir, str(i) + '_image_iter.png')))
        else:
            save_image(recon_img,
                       str(os.path.join(output_dir, str(i) + '_image_iter.png')))


################################################ NEW cub #########################################
def save_recon_cub_ae(iters, data_loader, enc, dec, cuda, output_dir_trvsl, fixed_idxs=[0]):
    output_dir_trvsl = '../output/' + output_dir_trvsl

    imgs = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        imgs[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            imgs[i] = imgs[i].cuda()
            imgs[i] = imgs[i].squeeze(0)
    imgs = torch.stack(imgs, dim=0)

    # do traversal and collect generated images

    feat = enc(imgs, num_samples=1)

    recon_img = dec.forward(feat)

    # save the generated files, also the animated gifs

    out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs))
    mkdirs(out_dir)
    save_image(imgs,
               str(os.path.join(out_dir, 'gt_image.png')), nrow=int(np.sqrt(imgs.shape[0])))

    save_image(recon_img,
               str(os.path.join(out_dir, 'recon_image.png')), nrow=int(np.sqrt(recon_img.shape[0])))


def save_recon_cub_vae(iters, data_loader, enc, dec, cuda, output_dir_trvsl, fixed_idxs=[0]):
    output_dir_trvsl = '../output/' + output_dir_trvsl

    imgs = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        imgs[i], _ = data_loader.dataset.__getitem__(idx)[:2]
        if cuda:
            imgs[i] = imgs[i].cuda()
            imgs[i] = imgs[i].squeeze(0)
    imgs = torch.stack(imgs, dim=0)

    # do traversal and collect generated images

    sample, _, _ = enc(imgs, num_samples=1)

    recon_img = dec.forward(sample)

    # save the generated files, also the animated gifs

    out_dir = os.path.join(output_dir_trvsl, str(iters), str(fixed_idxs))
    mkdirs(out_dir)
    save_image(imgs,
               str(os.path.join(out_dir, 'gt_image.png')), nrow=int(np.sqrt(imgs.shape[0])))

    save_image(recon_img,
               str(os.path.join(out_dir, 'recon_image.png')), nrow=int(np.sqrt(recon_img.shape[0])))
