from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import EncoderA, EncoderB, DecoderA, DecoderB
from datasets import datasets

import sys

sys.path.append('../')
import probtorch
import util
import visdom

# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=10, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=3,
                        help='size of the latent embedding of shared visual')
    parser.add_argument('--n_privateA', type=int, default=50,
                        help='size of the latent embedding of privateA')
    parser.add_argument('--n_privateB', type=int, default=82,
                        help='size of the latent embedding of privateB')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

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
    args = parser.parse_args()

# ------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'awa-run_id%d-privA%02ddim-privB%02ddim-lamb_text%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s-lr%s' % (
    args.run_id, args.n_privateA, args.n_privateB, args.lambda_text, args.beta1, args.beta2, args.seed,
    args.batch_size, args.wseed, args.lr)

DATA_PATH = '../data'

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA1 = (1., args.beta1, 1.)
BETA2 = (1., args.beta2, 1.)
N_LABELS = 50
N_ATTR = 85
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
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['test_acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


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

    llA = torch.tensor(np.stack([recon_A, recon_poeA, recon_crA], -1))
    llB = torch.tensor(np.stack([recon_B, recon_poeB, recon_crB], -1))
    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))

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


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', test_acc='win_test_acc', total_losses='win_total_losses'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB', 'recon_crA', 'recon_crB',
        'total_loss', 'test_total_loss', 'test_acc'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

path = '../../data/awa/Animals_with_Attributes2/'
test_classes = np.genfromtxt(path + 'testclasses.txt', delimiter='\n', dtype=str)
class_meta = np.genfromtxt(path + 'classes.txt', delimiter='\n', dtype=str)
test_labels = []
for test_class in test_classes:
    for i in range(len(class_meta)):
        if test_class in class_meta[i]:
            test_labels.append(i + 1)

train_data = torch.utils.data.DataLoader(datasets(train=True), batch_size=args.batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(datasets(train=False), batch_size=args.batch_size, shuffle=True)

BIAS_TRAIN = (train_data.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_data.dataset.__len__() - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zPrivate_dim=args.n_privateA, zShared_dim=args.n_shared)
decA = DecoderA(args.wseed, zPrivate_dim=args.n_privateA, zShared_dim=args.n_shared)
encB = EncoderB(args.wseed, zPrivate_dim=args.n_privateB, zShared_dim=args.n_shared)
decB = DecoderB(args.wseed, zPrivate_dim=args.n_privateB, zShared_dim=args.n_shared)
if CUDA:
    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)

optimizer = torch.optim.Adam(
    list(encB.parameters()) + list(decB.parameters()) + list(encA.parameters()) + list(decA.parameters()),
    lr=args.lr)


def elbo(q, pA, pB, lamb=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0):
    # from each of modality
    sharedA = []
    sharedB = []
    poe = []
    for i in range(args.n_shared):
        sharedA.append('sharedA' + str(i))
        sharedB.append('sharedB' + str(i))
        poe.append('poe' + str(i))

    privateB = []
    for i in range(args.n_privateB):
        privateB.append('privateB' + str(i))
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedA'],
                                                               latents=np.concatenate([['privateA'], sharedA]),
                                                               sample_dim=0,
                                                               batch_dim=1,
                                                               beta=beta1, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_sharedB'],
                                                               latents=np.concatenate([privateB, sharedB]),
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta2, bias=bias)

    if q['poe'] is not None:
        # by POE
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'],
                                                                         latents=np.concatenate([['privateA'], poe]),
                                                                         sample_dim=0,
                                                                         batch_dim=1,
                                                                         beta=beta1, bias=bias)
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_poe'],
                                                                         latents=np.concatenate([privateB, poe]),
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=beta2, bias=bias)

        # # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedB'],
                                                                       latents=np.concatenate([['privateA'], sharedB]),
                                                                       sample_dim=0,
                                                                       batch_dim=1,
                                                                       beta=beta1, bias=bias)
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_sharedA'],
                                                                       latents=np.concatenate([privateB, sharedA]),
                                                                       sample_dim=0, batch_dim=1,
                                                                       beta=beta2, bias=bias)

        loss = (reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B) + \
               (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) + \
               (reconst_loss_crA - kl_crA) + (lamb * reconst_loss_crB - kl_crB)

    else:
        reconst_loss_poeA = reconst_loss_crA = reconst_loss_poeB = reconst_loss_crB = None
        loss = 3 * ((reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B))
    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
                                                                          reconst_loss_crB]


def train(data, encA, decA, encB, decB, optimizer):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    pair_cnt = 0
    encA.train()
    encB.train()
    decA.train()
    decB.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)
    for b, (images, attr, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            # images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, N_LABELS)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)

            attr_onehot = torch.zeros(args.batch_size, N_ATTR, 2)
            attr_onehot.scatter_(2, attr.type(torch.int64).unsqueeze(2), 1)
            attr_onehot = torch.clamp(attr_onehot, EPS, 1 - EPS)  # 100, 85,2
            optimizer.zero_grad()
            # for test set, image modality is not trained
            if labels[0] in test_labels:
                if CUDA:
                    labels_onehot = labels_onehot.cuda()
                    attr_onehot = attr_onehot.cuda()

                # encode
                q = encA(images, num_samples=NUM_SAMPLES)
                q = encB(attr_onehot, num_samples=NUM_SAMPLES, q=q)
                ## poe ##
                for i in range(args.n_shared):
                    prior_logit = torch.zeros_like(
                        q['sharedA' + str(i)].dist.logits)  # prior is the concrete dist. of uniform dist.
                    poe_logit = q['sharedA' + str(i)].dist.logits + q['sharedB' + str(i)].dist.logits + prior_logit
                    q.concrete(logits=poe_logit,
                               temperature=TEMP,
                               name='poe' + str(i))
                # decode
                pA = decA(images, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                          num_samples=NUM_SAMPLES)
                pB = decB(labels_onehot, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                          num_samples=NUM_SAMPLES)
                # loss
                loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)
            else:
                if CUDA:
                    images = images.cuda()
                    labels_onehot = labels_onehot.cuda()
                    attr = attr.cuda()
                # encode
                q = encA(images, num_samples=NUM_SAMPLES)
                q = encB(attr, num_samples=NUM_SAMPLES, q=q)
                # q = encC(labels_onehot, num_samples=NUM_SAMPLES, q=q)
                ## poe ##
                for i in range(args.n_shared):
                    prior_logit = torch.zeros_like(
                        q['sharedA' + str(i)].dist.logits)  # prior is the concrete dist. of uniform dist.
                    poe_logit = q['sharedA' + str(i)].dist.logits + q['sharedB' + str(i)].dist.logits + prior_logit
                    # poe_logit = q['sharedA' + str(i)].dist.logits + q['sharedB' + str(i)].dist.logits + q['sharedC' + str(i)].dist.logits + prior_logit
                    q.concrete(logits=poe_logit,
                               temperature=TEMP,
                               name='poe' + str(i))
                # decode
                shared_dist = {'sharedA': [], 'sharedB': [], 'poe': []}
                for i in range(args.n_shared):
                    shared_dist['sharedA'].append(q['sharedA' + str(i)])
                    shared_dist['sharedB'].append(q['sharedB' + str(i)])
                    shared_dist['poe'].append(q['poe' + str(i)])
                pB = decB(attr, shared_dist, q=q,
                          num_samples=NUM_SAMPLES)
                pA = decA(images, shared_dist, q=q,
                          num_samples=NUM_SAMPLES)

                # loss
                loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)

            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
                recA[0] = recA[0].cpu()
                recB[0] = recB[0].cpu()

            epoch_elbo += loss.item()
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
                                                                                                   epoch_rec_crB / pair_cnt]


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
            N += 1
            # images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, args.n_shared)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
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

            batch_elbo, _, _ = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TEST)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            epoch_correct += pB['labels_sharedA'].loss.sum().item()

    if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
        util.evaluation.save_traverse(epoch, test_data, encA, decA, CUDA,
                                      output_dir_trvsl=MODEL_NAME, flatten_pixel=NUM_PIXELS,
                                      fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50])
        util.evaluation.save_reconst(epoch, test_data, encA, decA, encB, decB, CUDA,
                                     fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50], output_dir_trvsl=MODEL_NAME,
                                     flatten_pixel=NUM_PIXELS)

        save_ckpt(e + 1)
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
        label_idx.update({i: []})
    for idx in range(len(tr_labels)):
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

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB = train(train_data, encA, decA, encB, decB,
                                             optimizer)
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
    util.evaluation.save_reconst(args.epochs, test_data, encA, decA, encB, decB, CUDA,
                                 fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50], output_dir_trvsl=MODEL_NAME,
                                 flatten_pixel=NUM_PIXELS)
    util.evaluation.save_traverse(args.epochs, test_data, encA, decA, CUDA,
                                  fixed_idxs=[21, 2, 1, 10, 14, 25, 17, 86, 9, 50], output_dir_trvsl=MODEL_NAME,
                                  flatten_pixel=NUM_PIXELS)
    # util.evaluation.mutual_info(test_data, encA, CUDA, flatten_pixel=NUM_PIXELS)
else:
    save_ckpt(args.epochs)
