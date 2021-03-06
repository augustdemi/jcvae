import torch
import numpy as np
import os
from torchvision.utils import save_image

import sys
sys.path.append('../')
from probtorch.util import grid2gif, mkdirs
NUM_PIXELS = 784

def save_traverse(iters, data_loader, enc, dec, cuda, loc=-1):

    output_dir_trvsl = '../output/'
    fixed_idxs = [3, 2, 1, 18, 4, 15, 11, 17, 61, 99]
    out_dir = os.path.join(output_dir_trvsl, str(iters), 'testA')


    fixed_XA = [0] * len(fixed_idxs)
    # label = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        fixed_img = data_loader.dataset.__getitem__(idx)[0]
        fixed_XA[i] = fixed_img.view(-1, NUM_PIXELS)
        if cuda:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)

    fixed_XA = torch.stack(fixed_XA, dim=0)
    # cnt = [0] * 10
    # for l in label:
    #     cnt[l] += 1
    # print('cnt of digit:')
    # print(cnt)


    ####



    # do traversal and collect generated images


    q = enc(fixed_XA, num_samples=1)
    zA_ori, zS_ori = q['privateA'].dist.loc, q['sharedA'].value

    zA_dim = zA_ori.shape[2]
    zS_dim = zS_ori.shape[2]
    interpolation = torch.tensor(np.linspace(-2.5, 2.5, zS_dim))
    # saving_shape = torch.cat([fixed_XA[i] for i in range(fixed_XA.shape[0])], dim=0).shape
    # WS = torch.ones(saving_shape)
    # if cuda:
    #     WS = WS.cuda()

    tempA = []  # zA_dim + zS_dim , num_trv, 1, 32*num_samples, 32
    for row in range(zA_dim):
        if loc != -1 and row != loc:
            continue
        zA = zA_ori.clone()

        temp = []
        for val in interpolation:
            zA[:,:, row] = val

            hiddens = dec.dec_hidden(torch.cat([zA, zS_ori], -1))
            sampleA = dec.dec_image(hiddens).data
            sampleA = sampleA.view(sampleA.shape[0], -1, 28, 28)
            sampleA = torch.transpose(sampleA, 0,1)
            temp.append((torch.cat([sampleA[i] for i in range(sampleA.shape[0])], dim=1)).unsqueeze(0))

        tempA.append(torch.cat(temp, dim=0).unsqueeze(0))  # torch.cat(temp, dim=0) = num_trv, 1, 32*num_samples, 32

    temp = []
    for i in range(zS_dim):
        zS = np.zeros((1, 1, zS_dim))
        zS[0, 0 , i % zS_dim] = 1.
        zS = torch.Tensor(zS)
        zS = torch.cat([zS] * len(fixed_idxs), dim=1)

        if cuda:
            zS = zS.cuda()

        hiddens = dec.dec_hidden(torch.cat([zA_ori, zS], -1))
        sampleA = dec.dec_image(hiddens).data
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
        out_dir, str(os.path.join(out_dir, 'mnist_traverse' + '.gif')), delay=10
    )
