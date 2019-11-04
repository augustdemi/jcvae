import torch
import numpy as np
import os
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
# from dataset import CustomDataset

import sys
sys.path.append('../')
from probtorch.util import grid2gif, mkdirs, apply_poe


def save_traverse(iters, data_loader, enc, dec, cuda, fixed_idxs, output_dir_trvsl, flatten_pixel=None):

    output_dir_trvsl = '../output/' + output_dir_trvsl
    out_dir = os.path.join(output_dir_trvsl, str(iters))

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
    interpolation = torch.tensor(np.linspace(-2.5, 2.5, zS_dim))

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
    batch_size = q['privateA'].value.shape[1]
    z_private= q['privateA'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    z_shared= q['sharedA'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)

    q_ziCx_private = torch.exp(q['privateA'].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
    q_ziCx_shared = torch.exp(q['sharedA'].dist.log_pmf(z_shared).transpose(1, batch_dim + 1))
    q_ziCx = torch.cat((q_ziCx_private,q_ziCx_shared), dim=2)


    # for baseline
    # batch_size = q['styles'].value.shape[1]
    # z_private= q['styles'].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
    # q_ziCx_private = torch.exp(q['styles'].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
    # q_ziCx = q_ziCx_private


    latent_dim = q_ziCx.shape[-1]
    mi_zi_y = torch.tensor([.0] * latent_dim)
    for k in range(num_labels):
        q_ziCxk = q_ziCx[k * per_label_samples:(k + 1) * per_label_samples, k * per_label_samples:(k + 1) * per_label_samples, :]
        marg_q_ziCxk = q_ziCxk.sum(1)
        mi_zi_y += (marg_q_ziCxk * (np.log(batch_size/num_labels) + torch.log(marg_q_ziCxk) - torch.log(q_ziCx[k * per_label_samples:(k + 1) * per_label_samples, :, :].sum(1)))).mean(0)
    mi_zi_y = mi_zi_y / batch_size
    print(mi_zi_y)


    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.bar(range(latent_dim), mi_zi_y.detach().cpu().numpy())
    ax.set_xticks(range(latent_dim))
    ax.set_title('poeA')
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





class CustomDataset(Dataset):
    '''
    Dataset when it is possible and efficient to load all data items to memory
    '''

    ####
    def __init__(self, data_tensor, transform=None):
        '''
        data_tensor = actual data items; (N x C x H x W)
        '''

        self.data_tensor = data_tensor
        self.transform = transform

    ####
    def __getitem__(self, i):
        img = self.data_tensor[i]

        if self.transform is not None:
            img = self.transform(img)

        return img, i

    ####
    def __len__(self):
        return self.data_tensor.size(0)

def eval_disentangle_metric1(data_loader, cuda, encA, encB, zA_dim, zB_dim, zS_dim, NUM_PIXELS=4096):
    # some hyperparams
    num_pairs = 800  # # data pairs (d,y) for majority vote classification
    bs = 50  # batch size
    nsamps_per_factor = 100  # samples per factor
    nsamps_agn_factor = 5000  # factor-agnostic samples

    from torch.utils.data import DataLoader


    latent_classes, latent_values = np.load('../../data/3dfaces/gt_factor_labels.npy')
    latent_values = latent_values
    # latent values (actual values);(127050 x 4)
    latent_classes = latent_classes
    # classes ({0,1,...,K}-valued); (127050 x 4)
    latent_sizes = np.array([50, 21, 11, 11])
    N = latent_values.shape[0]

    dl = DataLoader(
        data_loader.dataset, batch_size=bs,
        shuffle=True, pin_memory=True)

    # root = os.path.join('../../data/3dfaces/basel_face_renders.pth')
    # data = torch.load(root).float().div(255)  # (50x21x11x11x64x64)
    # data = data.view(-1, 64, 64).unsqueeze(1)  # (127050 x 1 x 64 x 64)
    # train_kwargs = {'data_tensor': data}
    # dataset = CustomDataset(**train_kwargs)
    # dl = DataLoader( dataset, batch_size=bs, shuffle=True,
    #     num_workers=1, pin_memory=True)

    iterator = iter(dl)

    M = []
    for ib in range(int(nsamps_agn_factor / bs)):
        # sample a mini-batch
        _, XBb, _ = next(iterator)  # (bs x C x H x W)

        XBb = XBb.view(-1, NUM_PIXELS)
        if cuda:
            XBb = XBb.cuda()

        # encode
        q = encB(XBb, num_samples=1)
        mub = torch.cat([q['privateB'].dist.loc.squeeze(0), q['sharedB'].dist.loc.squeeze(0)], dim=1)
        M.append(mub.cpu().detach().numpy())
    M = np.concatenate(M, 0)

    # estimate sample vairance and mean of latent points for each dim
    vars_agn_factor = np.var(M, 0)

    # 2) estimatet dim-wise vars of latent points with "one factor fixed"

    factor_ids = range(0, len(latent_sizes))  # true factor ids
    vars_per_factor = np.zeros(
        [num_pairs, zB_dim + zS_dim])
    true_factor_ids = np.zeros(num_pairs, np.int)  # true factor ids

    # prepare data pairs for majority-vote classification
    i = 0
    for j in factor_ids:  # for each factor

        # repeat num_paris/num_factors times
        for r in range(int(num_pairs / len(factor_ids))):
            # a true factor (id and class value) to fix
            fac_id = j
            fac_class = np.random.randint(latent_sizes[fac_id])

            # randomly select images (with the fixed factor)
            indices = np.where(latent_classes[:, fac_id] == fac_class)[0]
            used_indices = dl.dataset.b_idx
            indices = np.array([elt for elt in indices if elt in used_indices])
            if len(indices) == 0:
                continue
            np.random.shuffle(indices)
            idx = indices[:nsamps_per_factor]
            M = []
            for ib in range(int(nsamps_per_factor / bs)):
                XBb = dl.dataset.get_3dface([idx[(ib * bs):(ib + 1) * bs]])
                # XBb, _ = dl.dataset[idx[(ib * bs):(ib + 1) * bs]]
                if XBb.shape[0] < 1:  # no more samples
                    continue;
                XBb = XBb.view(-1, NUM_PIXELS)
                if cuda:
                    XBb = XBb.cuda()
                q = encB(XBb, num_samples=1)
                mub = torch.cat([q['privateB'].dist.loc.squeeze(0), q['sharedB'].dist.loc.squeeze(0)], dim=1)
                M.append(mub.cpu().detach().numpy())
            M = np.concatenate(M, 0)

            # estimate sample var and mean of latent points for each dim
            if M.shape[0] >= 2:
                vars_per_factor[i, :] = np.var(M, 0)
            else:  # not enough samples to estimate variance
                vars_per_factor[i, :] = 0.0

                # true factor id (will become the class label)
            true_factor_ids[i] = fac_id

            i += 1

    # 3) evaluate majority vote classification accuracy

    # inputs in the paired data for classification
    smallest_var_dims = np.argmin(
        vars_per_factor / (vars_agn_factor + 1e-20), axis=1)

    # contingency table
    C = np.zeros([zB_dim + zS_dim, len(factor_ids)])
    for i in range(num_pairs):
        C[smallest_var_dims[i], true_factor_ids[i]] += 1

    num_errs = 0  # # misclassifying errors of majority vote classifier
    for k in range(zB_dim + zS_dim):
        num_errs += np.sum(C[k, :]) - np.max(C[k, :])

    metric1 = (num_pairs - num_errs) / num_pairs  # metric = accuracy
    return metric1, C


