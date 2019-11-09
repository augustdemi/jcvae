import torch
import numpy as np
import os
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
# from dataset import CustomDataset

import sys
sys.path.append('../')
from probtorch.util import grid2gif, mkdirs, apply_poe, transform


def save_traverse(iters, data_loader, enc, dec, cuda, fixed_idxs, output_dir_trvsl, flatten_pixel=None):

    output_dir_trvsl = '../output/' + output_dir_trvsl
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
            sampleA = dec.forward2(zA, zS_ori)
            if flatten_pixel is not None:
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                sampleA = torch.transpose(sampleA, 0,1)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    temp = []
    for i in range(zS_dim):
        zS = np.zeros((1, 1, zS_dim))
        zS[0, 0, i % zS_dim] = 1.
        zS = torch.Tensor(zS)
        zS = torch.cat([zS] * len(fixed_idxs), dim=1)
        if cuda:
            zS = zS.cuda()
        sampleA = dec.forward2(zA_ori, zS)
        if flatten_pixel is not None:
            sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
            sampleA = torch.transpose(sampleA, 0, 1)
        temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

    tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
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

def save_traverse_both(iters, data_loader, encA, decA, encB, decB, cuda, output_dir_trvsl, flatten_pixel=None, fixed_idxs = [3246, 7001, 14305, 19000, 27444, 33100, 38000, 45231, 51000, 55121]):

    tr_range = 2
    out_dir = os.path.join('../output/' + output_dir_trvsl, str(iters) +'_'+ str(-tr_range) + '~' + str(tr_range))

    fixed_XA = [0] * len(fixed_idxs)
    fixed_XB = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):

        fixed_XA[i], fixed_XB[i] = \
            data_loader.dataset.__getitem__(idx)[0:2]
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
            fixed_XB[i] = fixed_XB[i].cuda()

    fixed_XA = torch.stack(fixed_XA, dim=0)
    fixed_XB = torch.stack(fixed_XB, dim=0)

    # do traversal and collect generated images

    q = encA(fixed_XA, num_samples=1)
    zA_ori = q['privateA'].dist.loc
    q = encB(fixed_XB, num_samples=1, q=q)
    zB_ori = q['privateB'].dist.loc

    zS_ori, _ = apply_poe(cuda, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                    q['sharedB'].dist.loc, q['sharedB'].dist.scale)


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
            sampleA = decA.forward2(zA, zS_ori)
            if flatten_pixel is not None:
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                sampleA = torch.transpose(sampleA, 0,1)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    # shared A
    for row in range(zS_dim):
        if loc != -1 and row != loc:
            continue
        zS = zS_ori.clone()
        temp = []
        for val in interpolation:
            zS[:, :, row] = val
            sampleA = decA.forward2(zA_ori, zS)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))
        tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))
    # shared B
    for row in range(zS_dim):
        if loc != -1 and row != loc:
            continue
        zS = zS_ori.clone()

        temp = []
        for val in interpolation:
            zS[:, :, row] = val
            sampleB = decB.forward2(zB_ori, zS)
            temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))
        tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))

    ###### B_private
    for row in range(zB_dim):
        if loc != -1 and row != loc:
            continue
        zB = zB_ori.clone()

        temp = []
        for val in interpolation:
            zB[:, :, row] = val
            sampleB = decB.forward2(zB, zS_ori)
            if flatten_pixel is not None:
                sampleB = sampleB.view(sampleB.shape[0], -1, 28, 28)
                sampleB = torch.transpose(sampleB, 0,1)
            temp.append((torch.cat([sampleB[i] for i in range(sampleB.shape[0])], dim=1)).unsqueeze(0))
        tempAll.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    gifs = torch.cat(tempAll, dim=0)  # torch.Size([11, 10, 1, 384, 32])

    # save the generated files, also the animated gifs

    mkdirs(output_dir_trvsl)
    mkdirs(out_dir)

    for j, val in enumerate(interpolation):
        # I = torch.cat([IMG[key], gifs[:][j]], dim=0)
        I = gifs[:, j]
        save_image(
            tensor=I.cpu(),
            filename=os.path.join(out_dir, '%03d.jpg' % (j)),
            nrow=zA_dim + 2*zS_dim + zB_dim,
            pad_value=1)
        # make animated gif
    grid2gif(
        out_dir, str(os.path.join(out_dir, 'traverse.gif')), delay=10
    )



def save_traverse_base(iters, data_loader, enc, dec, cuda, fixed_idxs, output_dir_trvsl, flatten_pixel=None):

    output_dir_trvsl = '../output/' + output_dir_trvsl
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
            sampleA = dec.forward2(q['digits'].value, zA)
            if flatten_pixel is not None:
                sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
                sampleA = torch.transpose(sampleA, 0,1)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    temp = []
    for i in range(zS_dim):
        zS = np.zeros((1, 1, zS_dim))
        zS[0, 0, i % zS_dim] = 1.
        zS = torch.Tensor(zS)
        zS = torch.cat([zS] * len(fixed_idxs), dim=1)
        if cuda:
            zS = zS.cuda()
        sampleA = dec.forward2(zS, zA_ori)
        if flatten_pixel is not None:
            sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
            sampleA = torch.transpose(sampleA, 0, 1)
        temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

    tempA.append(torch.cat(temp, dim=0).unsqueeze(0))
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



def mutual_info(data_loader, enc, cuda, flatten_pixel=None):
    # fixed_idxs = [3, 2, 1, 18, 4, 15, 11, 17, 61, 99]

    num_labels = 10
    per_label_samples = 100
    per_label_cnt = {}
    for i in range(num_labels):
        per_label_cnt.update({i: 0})

    # fixed_XA = [0] * (num_labels * per_label_samples)
    # fixed_XB = [0] * (num_labels * per_label_samples)
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

    # for my model
    # batch_size = q['privateA'].value.shape[1]
    # z_private= q['privateA'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    # z_shared= q['sharedA'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    # q_ziCx_private = torch.exp(q['privateA'].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
    # q_ziCx_shared = torch.exp(q['sharedA'].dist.log_pmf(z_shared).transpose(1, batch_dim + 1))
    # q_ziCx = torch.cat((q_ziCx_private,q_ziCx_shared), dim=2)


    # for baseline
    # batch_size = q['styles'].value.shape[1]
    # z_private= q['styles'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    # q_ziCx_private = torch.exp(q['styles'].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
    # q_ziCx = q_ziCx_private


    # for baseline2
    batch_size = q['styles'].value.shape[1]
    z_private= q['styles'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    z_shared= q['digits'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    q_ziCx_private = torch.exp(q['styles'].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
    q_ziCx_shared = torch.exp(q['digits'].dist.log_pmf(z_shared).transpose(1, batch_dim + 1))
    q_ziCx = torch.cat((q_ziCx_private,q_ziCx_shared), dim=2)

    latent_dim = q_ziCx.shape[-1]
    mi_zi_y = torch.tensor([.0] * latent_dim)
    for k in range(num_labels):
        q_ziCxk = q_ziCx[k * per_label_samples:(k + 1) * per_label_samples, k * per_label_samples:(k + 1) * per_label_samples, :]
        marg_q_ziCxk = q_ziCxk.sum(1)
        mi_zi_y += (marg_q_ziCxk * (np.log(batch_size/num_labels) + torch.log(marg_q_ziCxk) - torch.log(q_ziCx[k * per_label_samples:(k + 1) * per_label_samples, :, :].sum(1)))).mean(0)
    mi_zi_y = mi_zi_y / batch_size
    print(mi_zi_y)


    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_subplot(111)
    ax.bar(range(latent_dim), mi_zi_y.detach().cpu().numpy())
    # ax.set_xticks(range(latent_dim))
    my_xticks = []
    for i in range(50):
        my_xticks.append('z' + str(i))
    my_xticks.append('c')
    plt.xticks(range(latent_dim), my_xticks)
    # ax.set_title('poeA')
    plt.show()

    # ax = fig.add_subplot(222)
    # ax.bar(range(11),poeB)
    # ax.set_xticks(range(11))
    # ax.set_title('poeB')
    #
    # ax = fig.add_subplot(223)
    # ax.bar(range(11),a)
    # ax.set_xticks(range(11))
    # ax.set_title('infA')
    #
    # ax = fig.add_subplot(224)
    # ax.bar(range(11),b)
    # ax.set_xticks(range(11))
    # ax.set_title('infB')



