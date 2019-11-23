from torchvision import datasets, transforms

import time
import random
import torch
import os
import visdom
import numpy as np

from model import EncoderA, EncoderB, DecoderA, DecoderB, EncoderC, DecoderC
from datasets import DIGIT

import sys
sys.path.append('../')
import probtorch
import util



#------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=10,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_private', type=int, default=10,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=75, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=75, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.0,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=50.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=5.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=10.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0,
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/svhn_mnist',
                        help='save and load path for ckpt')

    # visdom
    parser.add_argument('--viz_on',
                        default=False, type=probtorch.util.str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port',
                        default=8002, type=int, help='visdom port number')

    args = parser.parse_args()

#------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
# MODEL_NAME = 'svhn_mnist-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta%s-seed%s-lr%s-bs%s' % (args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta1, args.seed, args.lr, args.batch_size)
MODEL_NAME = 'svhn_mnist-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1_%s-beta2_%s-seed%s-lr%s-bs%s-wseed%s' % (
    args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta1, args.beta2, args.seed,
    args.lr, args.batch_size, args.wseed)
DATA_PATH = '../data'

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA1 = (1., args.beta1, 1.)
BETA2 = (1., args.beta2, 1.)


NUM_PIXELS = 784
TEMP = 0.66

NUM_SAMPLES = 1
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['test_acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['test_mi'])

def visualize_line():
    data = LINE_GATHER.data
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])

    recon_poeA = torch.Tensor(data['recon_poeA'])
    recon_poeB = torch.Tensor(data['recon_poeB'])
    recon_crA = torch.Tensor(data['recon_crA'])
    recon_crB = torch.Tensor(data['recon_crB'])
    total_loss = torch.Tensor(data['total_loss'])

    epoch = torch.Tensor(data['epoch'])
    test_acc = torch.Tensor(data['test_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])

    test_miA = torch.Tensor(data['test_miA'])
    test_miB = torch.Tensor(data['test_miB'])

    llA = torch.tensor(np.stack([recon_A, recon_poeA, recon_crA], -1))
    llB = torch.tensor(np.stack([recon_B, recon_poeB, recon_crB], -1))
    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))
    test_mi = torch.tensor(np.stack([test_miA, test_miB], -1))

    VIZ.line(
        X=epoch, Y=llA, env=MODEL_NAME + '/lines',
        win=WIN_ID['llA'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalA', legend=['A', 'poeA', 'crA'])
    )
    VIZ.line(
        X=epoch, Y=llB, env=MODEL_NAME + '/lines',
        win=WIN_ID['llB'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalB', legend=['B', 'poeB', 'crB'])
    )

    VIZ.line(
        X=epoch, Y=test_acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['test_acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Test Accuracy', legend=['acc'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )

    VIZ.line(
        X=epoch, Y=test_mi, env=MODEL_NAME + '/lines',
        win=WIN_ID['test_mi'], update='append',
        opts=dict(xlabel='epoch', ylabel='mi',
                  title='Test normalized MI(c,y)', legend=['svhn', 'mnist'])
    )

if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', test_acc='win_test_acc', total_losses='win_total_losses', test_mi='win_test_mi'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB', 'recon_crA', 'recon_crB',
        'total_loss', 'test_total_loss', 'test_acc', 'test_miA', 'test_miB'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()



dset = DIGIT('./data', train=True)
train_data = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(DIGIT('./data', train=False), batch_size=args.batch_size, shuffle=True)

BIAS_TRAIN = (train_data.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_data.dataset.__len__() - 1) / (args.batch_size - 1)

TRAIN_ITER_PER_EPO = train_data.dataset.__len__() / args.batch_size
TEST_ITER_PER_EPO = test_data.dataset.__len__() / args.batch_size

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zPrivate_dim=args.n_private)
decA = DecoderA(args.wseed, zPrivate_dim=args.n_private)
encB = EncoderB(args.wseed, zPrivate_dim=args.n_private)
decB = DecoderB(args.wseed, zPrivate_dim=args.n_private)
if CUDA:
    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)

optimizer =  torch.optim.Adam(list(encB.parameters())+list(decB.parameters())+list(encA.parameters())+list(decA.parameters()),
                              lr=args.lr)


def mutual_info(data_loader, encA, encB, cuda, flatten_pixel=None, plot=False):
    num_labels = 10
    per_label_samples = 100
    per_label_cnt = {}
    for i in range(num_labels):
        per_label_cnt.update({i: 0})

    fixed_XA = []
    fixed_XB = []
    fixed_XC = []
    for i in range(num_labels):
        j = 0
        while per_label_cnt[i] < per_label_samples:
            imgA, imgB, label = data_loader.dataset.__getitem__(j)
            if label == i:
                if flatten_pixel is not None:
                    imgB = imgB.view(-1, flatten_pixel)
                if cuda:
                    imgA = imgA.cuda()
                    imgB = imgB.cuda()
                imgA = imgA.squeeze(0)
                imgB = imgB.squeeze(0)
                fixed_XA.append(imgA)
                fixed_XB.append(imgB)
                fixed_XC.append(label)
                per_label_cnt[i] += 1
            j += 1

    fixed_XA = torch.stack(fixed_XA, dim=0)
    fixed_XB = torch.stack(fixed_XB, dim=0)
    batch_dim = 1
    mi = []
    all_latent_var = [['privateA', 'sharedA'], ['privateB', 'sharedB']]

    for i in range(2):
        latents = all_latent_var[i]
        if i == 0:
            q = encA(fixed_XA, num_samples=1)
        else:
            q = encB(fixed_XB, num_samples=1)
        batch_size = q[latents[0]].value.shape[1]
        z_private = q[latents[0]].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
        z_shared = q[latents[1]].value.unsqueeze(batch_dim + 1).transpose(batch_dim, 0)
        q_ziCx_private = torch.exp(q[latents[0]].dist.log_prob(z_private).transpose(1, batch_dim + 1).squeeze(2))
        q_ziCx_shared = torch.exp(q[latents[1]].dist.log_pmf(z_shared).transpose(1, batch_dim + 1))
        q_ziCx = torch.cat((q_ziCx_private, q_ziCx_shared), dim=2)

        latent_dim = q_ziCx.shape[-1]
        mi_zi_y = torch.tensor([.0] * latent_dim)
        if cuda:
            mi_zi_y = mi_zi_y.cuda()
        for k in range(num_labels):
            q_ziCxk = q_ziCx[k * per_label_samples:(k + 1) * per_label_samples,
                      k * per_label_samples:(k + 1) * per_label_samples, :]
            marg_q_ziCxk = q_ziCxk.sum(1)
            mi_zi_y += (marg_q_ziCxk * (np.log(batch_size / num_labels) + torch.log(marg_q_ziCxk) - torch.log(
                q_ziCx[k * per_label_samples:(k + 1) * per_label_samples, :, :].sum(1)))).mean(0)
        mi_zi_y = mi_zi_y / batch_size
        print(mi_zi_y)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(3, 2))
            ax = fig.add_subplot(111)
            ax.bar(range(latent_dim), mi_zi_y.detach().cpu().numpy())
            my_xticks = []
            for i in range(latent_dim - 1):
                my_xticks.append('z' + str(i + 1))
            my_xticks.append('c')
            plt.xticks(range(latent_dim), my_xticks)
            plt.show()
        mi.append(mi_zi_y.detach().cpu().numpy())
    return mi


def elbo(q, pA, pB, lamb=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0, train=True):
    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedA'], latents=['privateA', 'sharedA'], sample_dim=0, batch_dim=1,
                                        beta=beta1, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images_sharedB'], latents=['privateB', 'sharedB'], sample_dim=0, batch_dim=1,
                                        beta=beta2, bias=bias)

    if q['poe'] is not None:
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'], latents=['privateA', 'poe'], sample_dim=0, batch_dim=1,
                                                    beta=beta1, bias=bias)
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images_poe'], latents=['privateB', 'poe'], sample_dim=0, batch_dim=1,
                                                    beta=beta2, bias=bias)

        # # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedB'], latents=['privateA', 'sharedB'], sample_dim=0, batch_dim=1,
                                                    beta=beta1, bias=bias)
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images_sharedA'], latents=['privateB', 'sharedA'], sample_dim=0, batch_dim=1,
                                                    beta=beta2, bias=bias)

        loss = (reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B) + \
               (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) + \
               (reconst_loss_crA - kl_crA) + (lamb * reconst_loss_crB - kl_crB)
    else:
        reconst_loss_poeA = reconst_loss_crA = reconst_loss_poeB = reconst_loss_crB = None
        loss = 3 * ((reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B))
    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
                                                                          reconst_loss_crB]

def train(data, encA, decA, encB, decB, optimizer,
          label_mask={}, fixed_imgs=None, fixed_labels=None):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    pair_cnt = 0
    encA.train()
    encB.train()
    decB.train()
    decA.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)
    for b, (svhn, mnist, label) in enumerate(data):
        if svhn.size()[0] == args.batch_size:
            if args.label_frac > 1 and random.random() < args.sup_frac:
                N += 1
                mnist = mnist.view(-1, NUM_PIXELS)
                if CUDA:
                    svhn = svhn.cuda()
                    mnist = mnist.cuda()
                optimizer.zero_grad()
                if b not in label_mask:
                    label_mask[b] = (random.random() < args.label_frac)
                if (label_mask[b] and args.label_frac == args.sup_frac):
                    # encode
                    q = encA(svhn, num_samples=NUM_SAMPLES)
                    q = encB(mnist, num_samples=NUM_SAMPLES, q=q)
                    ## poe ##
                    prior_logit = torch.zeros_like(
                        q['sharedA'].dist.logits)  # prior is the concrete dist. of uniform dist.
                    poe_logit = q['sharedA'].dist.logits + q['sharedB'].dist.logits + prior_logit
                    q.concrete(logits=poe_logit,
                               temperature=TEMP,
                               name='poe')
                    # decode
                    pA = decA(svhn, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                              num_samples=NUM_SAMPLES)
                    pB = decB(mnist, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                              num_samples=NUM_SAMPLES)

                    for param in encB.parameters():
                        param.requires_grad = True
                    for param in decB.parameters():
                        param.requires_grad = True
                    # loss
                    loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)
            else:
                N += 1
                mnist = mnist.view(-1, NUM_PIXELS)
                if CUDA:
                    svhn = svhn.cuda()
                    mnist = mnist.cuda()
                optimizer.zero_grad()
                if b not in label_mask:
                    label_mask[b] = (random.random() < args.label_frac)
                if (label_mask[b] and args.label_frac == args.sup_frac):
                    # encode
                    q = encA(svhn, num_samples=NUM_SAMPLES)
                    q = encB(mnist, num_samples=NUM_SAMPLES, q=q)
                    ## poe ##
                    prior_logit = torch.zeros_like(q['sharedA'].dist.logits)  # prior is the concrete dist. of uniform dist.
                    poe_logit = q['sharedA'].dist.logits + q['sharedB'].dist.logits + prior_logit
                    q.concrete(logits=poe_logit,
                               temperature=TEMP,
                               name='poe')
                    # decode
                    pA = decA(svhn, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe':q['poe']}, q=q,
                            num_samples=NUM_SAMPLES)
                    pB = decB(mnist, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe':q['poe']}, q=q,
                            num_samples=NUM_SAMPLES)

                    for param in encB.parameters():
                        param.requires_grad = True
                    for param in decB.parameters():
                        param.requires_grad = True
                    # loss
                    loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)
                else:
                    shuffled_idx = list(range(args.batch_size))
                    random.shuffle(shuffled_idx)
                    mnist = mnist[shuffled_idx]

                    q = encA(svhn, num_samples=NUM_SAMPLES)
                    q = encB(mnist, num_samples=NUM_SAMPLES, q=q)
                    # decode
                    pA = decA(svhn, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                              num_samples=NUM_SAMPLES)
                    pB = decB(mnist, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                              num_samples=NUM_SAMPLES)
                    for param in encB.parameters():
                        param.requires_grad = False
                    for param in decB.parameters():
                        param.requires_grad = False
                    loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)

            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
                recA[0] = recA[0].cpu()
                recB[0] = recB[0].cpu()

            epoch_elbo -= loss.item()
            epoch_recA += recA[0].item()
            epoch_recB += recB[0].item()

            if recA[1] is not None:
                if CUDA:
                    for i in range(2):
                        recA[i] = recA[i].cpu()
                        recB[i] = recB[i].cpu()
                epoch_rec_poeA += recA[1].item()
                epoch_rec_crA += recA[2].item()
                epoch_rec_poeB += recB[1].item()
                epoch_rec_crB += recB[2].item()
                pair_cnt += 1

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / pair_cnt, epoch_rec_crA / pair_cnt], [epoch_recB / N,
                                                                                                   epoch_rec_poeB / pair_cnt,
                                                                                                   epoch_rec_crB / pair_cnt], label_mask

def test(data, encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0
    for b, (svhn, mnist, label) in enumerate(data):
        if svhn.size()[0] == args.batch_size:
            N += 1
            mnist = mnist.view(-1, NUM_PIXELS)
            if CUDA:
                svhn = svhn.cuda()
                mnist = mnist.cuda()
            # encode
            q = encA(svhn, num_samples=NUM_SAMPLES)
            q = encB(mnist, num_samples=NUM_SAMPLES, q=q)
            pA = decA(svhn, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(mnist, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)

            batch_elbo, _, _ = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TEST)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            epoch_correct += 0

    if (epoch+1) % 5 ==  0 or epoch+1 == args.epochs:
        util.evaluation.save_traverse_both(epoch, test_data, encA, decA, encB, decB, CUDA,
                                           output_dir_trvsl=MODEL_NAME, flatten_pixel=NUM_PIXELS,
                                           fixed_idxs=[0, 600, 10000, 12000, 16001, 18000, 19000, 21000, 23000, 25000])
        save_ckpt(e+1)
    return epoch_elbo / N, 1 + epoch_correct / (N * args.batch_size)


def save_ckpt(e):
    if not os.path.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    torch.save(encA.state_dict(),
               '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decA.state_dict(),
               '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(encB.state_dict(),
               '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decB.state_dict(),
               '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


def get_paired_data(paired_cnt, seed):
    data = torch.utils.data.DataLoader(
        datasets.SVHN(DATA_PATH, split='train', download=True,
                      transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)
    tr_labels = data.dataset.labels

    cnt = int(paired_cnt / 10)
    assert cnt == paired_cnt / 10

    label_idx = {}
    for i in range(10):
        label_idx.update({i:[]})
    for idx in  range(len(tr_labels)):
        label = tr_labels[idx]
        label_idx[label].append(idx)

    total_random_idx = []
    for i in range(10):
        random.seed(seed)
        per_label_random_idx = random.sample(label_idx[i], cnt)
        total_random_idx.extend(per_label_random_idx)
    random.seed(seed)
    random.shuffle(total_random_idx)

    imgs = []
    labels = []
    for idx in total_random_idx:
        img, label = data.dataset.__getitem__(idx)
        imgs.append(img)
        labels.append(torch.tensor(label))
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    return imgs, labels

if args.ckpt_epochs > 0:
    if CUDA:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        decA.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        encB.load_state_dict(torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        decB.load_state_dict(torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
    else:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))
        decA.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))
        encB.load_state_dict(torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))
        decB.load_state_dict(torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))

mask = {}
fixed_imgs=None
fixed_labels=None
if args.label_frac > 1:
    fixed_imgs, fixed_labels = get_paired_data(args.label_frac, args.seed)


for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB, mask = train(train_data, encA, decA, encB, decB,
                                                   optimizer, mask, fixed_imgs=fixed_imgs, fixed_labels=fixed_labels)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB, e)

    if args.viz_on:
        mi = mutual_info(test_data, encA, encB, CUDA, flatten_pixel=NUM_PIXELS, plot=False)
        miA = (mi[0] / np.linalg.norm(mi[0]))[args.n_private]
        miB = (mi[1] / np.linalg.norm(mi[1]))[args.n_private]

        LINE_GATHER.insert(epoch=e,
                           test_miA=miA,
                           test_miB=miB,
                           test_acc=test_accuracy,
                           test_total_loss=-test_elbo,
                           total_loss=-train_elbo,
                           recon_A=rec_lossA[0],
                           recon_poeA=rec_lossA[1],
                           recon_crA=rec_lossA[2],
                           recon_B=rec_lossB[0],
                           recon_poeB=rec_lossB[1],
                           recon_crB=rec_lossB[2],
                           )
        visualize_line()
        LINE_GATHER.flush()


    test_end = time.time()
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start,
            test_elbo, test_accuracy, test_end - test_start))


if args.ckpt_epochs == args.epochs:
    util.evaluation.save_traverse_both(args.epochs, test_data, encA, decA, encB, decB, CUDA,
                                       output_dir_trvsl=MODEL_NAME, flatten_pixel=NUM_PIXELS,
                                       fixed_idxs=[0, 6000, 10000, 12000, 16001, 18000, 19000, 21000, 23000, 25000])
    # util.evaluation.save_reconst(args.epochs, test_data, encA, decA, encB, decB, CUDA, fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50], output_dir_trvsl=MODEL_NAME)
    # util.evaluation.mutual_info(test_data, encA, CUDA, flatten_pixel=NUM_PIXELS)
else:
    save_ckpt(args.epochs)
