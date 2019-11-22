from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import EncoderA, EncoderB, DecoderA, DecoderB

import sys
sys.path.append('../')
import probtorch
import util
import visdom



#------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=10, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=10,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_private', type=int, default=50,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=200.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=3.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=3.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/svhn',
                        help='save and load path for ckpt')
    # visdom
    parser.add_argument('--viz_on',
                        default=False, type=probtorch.util.str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port',
                        default=8002, type=int, help='visdom port number')
    parser.add_argument('--viz_ll_iter',
                        default=100, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter',
                        default=100, type=int, help='visdom line data applying iter')
    args = parser.parse_args()

#------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'svhn-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s' % (
    args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta1, args.beta2, args.seed,
    args.batch_size, args.wseed)

DATA_PATH = '../data'

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA1 = (1., args.beta1, 1.)
BETA2 = (1., args.beta2, 1.)
# BIAS_TRAIN = 1.0
# BIAS_TEST = 1.0
# model parameters
# NUM_PIXELS = 3*32*32


NUM_PIXELS = None
TEMP = 0.66

NUM_SAMPLES = 1
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['recon'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['test_acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])
    # if self.eval_metrics:
    #     self.viz.close(env=self.name+'/lines', win=WIN_ID['metrics'])


####
def visualize_line():
    # prepare data to plot
    data = LINE_GATHER.data
    iters = torch.Tensor(data['iter'])
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])

    full_modal_iter = torch.Tensor(data['full_modal_iter'])
    recon_poeA = torch.Tensor(data['recon_poeA'])
    recon_poeB = torch.Tensor(data['recon_poeB'])
    recon_crA = torch.Tensor(data['recon_crA'])
    recon_crB = torch.Tensor(data['recon_crB'])
    total_loss = torch.Tensor(data['total_loss'])

    epoch = torch.Tensor(data['epoch'])
    test_acc = torch.Tensor(data['test_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])
    test_mi_given_y = torch.Tensor(data['test_mi'])

    recons = torch.stack(
        [recon_A.detach(), recon_B.detach()], -1
    )

    total_losses = torch.stack(
        [torch.tensor(total_loss), torch.tensor(test_total_loss)], -1
    )

    VIZ.line(
        X=iters, Y=recons, env=MODEL_NAME + '/lines',
        win=WIN_ID['recon'], update='append',
        opts=dict(xlabel='iter', ylabel='recon losses',
                  title='Train Losses', legend=['recon_A', 'recon_B'])
    )

    recons2 = torch.stack(
        [recon_poeA.detach(), recon_poeB.detach(), recon_crA.detach(), recon_crB.detach()], -1
    )
    VIZ.line(
        X=full_modal_iter, Y=recons2, env=MODEL_NAME + '/lines',
        win=WIN_ID['recon2'], update='append',
        opts=dict(xlabel='iter', ylabel='recon losses',
                  title='Train Losses - with full modal', legend=['recon_poeA', 'recon_poeB', 'recon_crA', 'recon_crB'])
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
        opts=dict(xlabel='epoch', ylabel='total loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )

    VIZ.line(
        X=epoch, Y=test_mi_given_y, env=MODEL_NAME + '/lines',
        win=WIN_ID['test_mi_given_y'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Test MI of concrete var given label', legend=['mi'])
    )

if args.viz_on:
    WIN_ID = dict(
        recon='win_recon', recon2='win_recon2', test_acc='win_test_acc', total_losses='win_total_losses',
        test_mi_given_y='test_mi_given_y'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'iter', 'epoch', 'full_modal_iter', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB', 'recon_crA', 'recon_crB',
        'total_loss', 'test_total_loss', 'test_acc', 'test_mi'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()


train_data = torch.utils.data.DataLoader(
                datasets.SVHN(DATA_PATH, split='train', download=True,
                               transform=transforms.ToTensor()),
                batch_size=args.batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(
                datasets.SVHN(DATA_PATH, split='test', download=True,
                               transform=transforms.ToTensor()),
                batch_size=args.batch_size, shuffle=True)

BIAS_TRAIN = (train_data.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_data.dataset.__len__() - 1) / (args.batch_size - 1)

TRAIN_ITER_PER_EPO = train_data.dataset.__len__() / args.batch_size

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zPrivate_dim=args.n_private)
decA = DecoderA(args.wseed, zPrivate_dim=args.n_private)
encB = EncoderB(args.wseed)
decB = DecoderB(args.wseed)
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


def elbo(epoch, iter, q, pA, pB, lamb=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0):
    iter += epoch * TRAIN_ITER_PER_EPO
    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedA'], latents=['privateA', 'sharedA'], sample_dim=0, batch_dim=1,
                                                               beta=beta1, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_sharedB'], latents=['sharedB'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta2, bias=bias)

    if q['poe'] is not None:
        # by POE
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'], latents=['privateA', 'poe'], sample_dim=0, batch_dim=1,
                                                                         beta=beta1, bias=bias)
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_poe'], latents=['poe'], sample_dim=0, batch_dim=1,
                                                                         beta=beta2, bias=bias)

        # # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedB'], latents=['privateA', 'sharedB'], sample_dim=0, batch_dim=1,
                                                                       beta=beta1, bias=bias)
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_sharedA'], latents=['sharedA'], sample_dim=0, batch_dim=1,
                                                                       beta=beta2, bias=bias)

        loss = (reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B) + \
               (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) + \
               (reconst_loss_crA - kl_crA) + (lamb * reconst_loss_crB - kl_crB)

        if args.viz_on and iter % args.viz_ll_iter == 0:
            LINE_GATHER.insert(full_modal_iter=(epoch * TRAIN_ITER_PER_EPO + iter),
                               recon_poeA=reconst_loss_poeA.item(),
                               recon_poeB=reconst_loss_poeB.item(),
                               recon_crA=reconst_loss_crA.item(),
                               recon_crB=reconst_loss_crB.item()
                               )
    else:
        loss = 3*((reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B))

    if args.viz_on and iter % args.viz_ll_iter == 0:
        LINE_GATHER.insert(iter=iter,
                           recon_A=reconst_loss_A.item(),
                           recon_B=reconst_loss_B.item()
                           )
    return loss


def train(data, encA, decA, encB, decB, optimizer, epoch,
          label_mask={}, fixed_imgs=None, fixed_labels=None):
    epoch_elbo = 0.0
    encA.train()
    encB.train()
    decB.train()
    decA.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            if args.label_frac > 1 and random.random() < args.sup_frac:
                N += args.batch_size
                shuffled_idx = list(range(int(args.label_frac)))
                random.shuffle(shuffled_idx)
                shuffled_idx = shuffled_idx[:args.batch_size]
                images = fixed_imgs[shuffled_idx]
                fixed_labels_batch = fixed_labels[shuffled_idx]

                labels_onehot = torch.zeros(args.batch_size, args.n_shared)
                labels_onehot.scatter_(1, fixed_labels_batch.unsqueeze(1), 1)
                labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
                optimizer.zero_grad()
                if CUDA:
                    images = images.cuda()
                    labels_onehot = labels_onehot.cuda()

                # encode
                q = encA(images, num_samples=NUM_SAMPLES)
                q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
                ## poe ##
                prior_logit = torch.zeros_like(q['sharedA'].dist.logits)  # prior is the concrete dist. of uniform dist.
                poe_logit = q['sharedA'].dist.logits + q['sharedB'].dist.logits + prior_logit
                q.concrete(logits=poe_logit,
                           temperature=TEMP,
                           name='poe')
                # decode
                pA = decA(images, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                          num_samples=NUM_SAMPLES)
                pB = decB(labels_onehot, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                          num_samples=NUM_SAMPLES)
                for param in encB.parameters():
                    param.requires_grad = True
                for param in decB.parameters():
                    param.requires_grad = True
                # loss
                loss = -elbo(epoch, b, q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)
            else:
                N += args.batch_size
                # images = images.view(-1, NUM_PIXELS)
                labels_onehot = torch.zeros(args.batch_size, args.n_shared)
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS)
                if CUDA:
                    images = images.cuda()
                    labels_onehot = labels_onehot.cuda()
                optimizer.zero_grad()
                if b not in label_mask:
                    label_mask[b] = (random.random() < args.label_frac)
                if (label_mask[b] and args.label_frac == args.sup_frac):
                    # encode
                    q = encA(images, num_samples=NUM_SAMPLES)
                    q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
                    ## poe ##
                    prior_logit = torch.zeros_like(q['sharedA'].dist.logits)  # prior is the concrete dist. of uniform dist.
                    poe_logit = q['sharedA'].dist.logits + q['sharedB'].dist.logits + prior_logit
                    q.concrete(logits=poe_logit,
                               temperature=TEMP,
                               name='poe')
                    # decode
                    pA = decA(images, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe':q['poe']}, q=q,
                            num_samples=NUM_SAMPLES)
                    pB = decB(labels_onehot, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe':q['poe']}, q=q,
                            num_samples=NUM_SAMPLES)

                    # if b % 10 ==0:
                    #     print('--------------------------------iter ', b, '---------------------------------------')
                    #     print('sharedA')
                    #     cnt = [0] * 10
                    #     for elt in q['sharedA'].value.argmax(dim=2)[0]:
                    #         cnt[elt] +=1
                    #     print(cnt)
                    #
                    #     print('poe')
                    #     # print(q['poe'].value.argmax(dim=2)[0][:20])
                    #     cnt = [0] * 10
                    #     for elt in q['poe'].value.argmax(dim=2)[0]:
                    #         cnt[elt] +=1
                    #     print(cnt)
                    #
                    #     print('sharedB')
                    #     # print(q['sharedB'].value.argmax(dim=2)[0][:20])
                    #     cnt = [0] * 10
                    #     for elt in q['sharedB'].value.argmax(dim=2)[0]:
                    #         cnt[elt] +=1
                    #     print(cnt)
                    #
                    #     print('labels')
                    #     cnt = [0] * 10
                    #     for elt in labels:
                    #         cnt[elt] +=1
                    #     print(cnt)

                    for param in encB.parameters():
                        param.requires_grad = True
                    for param in decB.parameters():
                        param.requires_grad = True
                    # loss
                    loss = -elbo(epoch, b, q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)
                else:
                    # labels_onehot = labels_onehot[:, torch.randperm(10)]
                    q = encA(images, num_samples=NUM_SAMPLES)
                    q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
                    pA = decA(images, {'sharedA': q['sharedA']}, q=q,
                              num_samples=NUM_SAMPLES)
                    pB = decB(labels_onehot, {'sharedB': q['sharedB']}, q=q,
                              num_samples=NUM_SAMPLES)
                    for param in encB.parameters():
                        param.requires_grad = False
                    for param in decB.parameters():
                        param.requires_grad = False
                    loss = -elbo(epoch, b, q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)

            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_elbo -= loss.item()

    return epoch_elbo / N, label_mask

def test(data, encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += args.batch_size
            # images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, args.n_shared)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
            pA = decA(images, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(labels_onehot, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES, train=False)

            batch_elbo = elbo(epoch, b, q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TEST)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            epoch_correct += pB['labels_sharedA'].loss.sum().item()

    if (epoch+1) % 5 ==  0 or epoch+1 == args.epochs:
        util.evaluation.save_traverse(epoch, test_data, encA, decA, CUDA,
                                           output_dir_trvsl=MODEL_NAME, flatten_pixel=NUM_PIXELS, fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50])
        util.evaluation.save_reconst(epoch, test_data, encA, decA, encB, decB, CUDA,
                                     fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50], output_dir_trvsl=MODEL_NAME,
                                     flatten_pixel=NUM_PIXELS)

        save_ckpt(e+1)
    return epoch_elbo / N, 1 + epoch_correct / N



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
    train_elbo, mask = train(train_data, encA, decA, encB, decB,
                             optimizer, e, mask, fixed_imgs=fixed_imgs, fixed_labels=fixed_labels)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB, e)

    mi = util.evaluation.mutual_info(test_data, encA, CUDA, flatten_pixel=NUM_PIXELS)
    mi = mi / np.linalg.norm(mi)

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_acc=test_accuracy,
                           test_total_loss=test_elbo,
                           total_loss=train_elbo,
                           test_mi=mi[args.n_private]
                           )
        visualize_line()
        LINE_GATHER.flush()

    test_end = time.time()
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start,
            test_elbo, test_accuracy, test_end - test_start))


if args.ckpt_epochs == args.epochs:
    util.evaluation.save_reconst(args.epochs, test_data, encA, decA, encB, decB, CUDA, fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50], output_dir_trvsl=MODEL_NAME, flatten_pixel=NUM_PIXELS)
    util.evaluation.save_traverse(args.epochs, test_data, encA, decA, CUDA,
                                  fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50], output_dir_trvsl=MODEL_NAME,
                                  flatten_pixel=NUM_PIXELS)
    # util.evaluation.mutual_info(test_data, encA, CUDA, flatten_pixel=NUM_PIXELS)
else:
    save_ckpt(args.epochs)
