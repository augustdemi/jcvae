from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import visdom
import numpy as np

from model import EncoderA, EncoderB, DecoderA, DecoderB
from datasets import datasets
from sklearn.metrics import f1_score

import sys

sys.path.append('../')
import probtorch
import util

# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=2, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=18,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_private', type=int, default=100,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=100.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=3.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/celeba_cont/1',
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
MODEL_NAME = 'celeba_cont-run_id%d-priv%02ddim-shared%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s-lr%s' % (
    args.run_id, args.n_private, args.n_shared, args.label_frac, args.sup_frac, args.lambda_text, args.beta1,
    args.beta2, args.seed,
    args.batch_size, args.wseed, args.lr)
DATA_PATH = '../data'
ATTR_TO_PLOT = ['Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Blond_Hair', 'Eyeglasses', 'Bangs', 'off',
                'Black_Hair', 'Wavy_Hair']

ATTR_TO_IX_DICT = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18,
                   'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38,
                   'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21,
                   'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26,
                   'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32,
                   'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22,
                   'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3,
                   'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23,
                   'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}
# we only keep 18 of the more visually distinctive features
# See [1] Perarnau, Guim, et al. "Invertible conditional gans for
#         image editing." arXiv preprint arXiv:1611.06355 (2016).
ATTR_IX_TO_KEEP = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
IX_TO_ATTR_DICT = {v: k for k, v in ATTR_TO_IX_DICT.items()}

if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA1 = (1., args.beta1, 1.)
BETA2 = (1., args.beta2, 1.)
# model parameters
NUM_PIXELS = 784
TEMP = 0.66
NUM_SAMPLES = 1
N_ATTR = 18


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['f1'])
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
    val_acc = torch.Tensor(data['val_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])
    val_total_loss = torch.Tensor(data['val_total_loss'])
    test_f1 = torch.Tensor(data['test_f1'])
    val_f1 = torch.Tensor(data['val_f1'])

    llA = torch.tensor(np.stack([recon_A, recon_poeA, recon_crA], -1))
    llB = torch.tensor(np.stack([recon_B, recon_poeB, recon_crB], -1))
    total_losses = torch.tensor(np.stack([total_loss, test_total_loss, val_total_loss], -1))
    acc = torch.tensor(np.stack([test_acc, val_acc], -1))
    f1 = torch.tensor(np.stack([test_f1, val_f1], -1))

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
        X=epoch, Y=acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Accuracy', legend=['test_acc', 'val_acc'])
    )

    VIZ.line(
        X=epoch, Y=f1, env=MODEL_NAME + '/lines',
        win=WIN_ID['f1'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='F1 score', legend=['test_f1', 'val_f1'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss', 'val_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', acc='win_acc', total_losses='win_total_losses', f1='win_f1'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB', 'recon_crA', 'recon_crB',
        'total_loss', 'test_total_loss', 'test_acc', 'val_total_loss', 'val_acc', 'test_f1', 'val_f1'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

preprocess_data = transforms.Compose([
    transforms.CenterCrop((168, 178)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_data = torch.utils.data.DataLoader(datasets(partition='train', data_dir='../../data/celeba2',
                                                  image_transform=preprocess_data), batch_size=args.batch_size,
                                         shuffle=False)

test_data = torch.utils.data.DataLoader(datasets(partition='test', data_dir='../../data/celeba2',
                                                 image_transform=preprocess_data), batch_size=args.batch_size,
                                        shuffle=False)
val_data = torch.utils.data.DataLoader(datasets(partition='val', data_dir='../../data/celeba2',
                                                image_transform=preprocess_data), batch_size=args.batch_size,
                                       shuffle=False)

print('>>> data loaded')
print('train: ', len(train_data.dataset))
print('val: ', len(val_data.dataset))
print('test: ', len(test_data.dataset))

BIAS_TRAIN = (len(train_data.dataset) - 1) / (args.batch_size - 1)
BIAS_VAL = (len(val_data.dataset) - 1) / (args.batch_size - 1)
BIAS_TEST = (len(test_data.dataset) - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zPrivate_dim=args.n_private, zShared_dim=args.n_shared)
decA = DecoderA(args.wseed, zPrivate_dim=args.n_private, zShared_dim=args.n_shared)
encB = EncoderB(args.wseed, num_attr=N_ATTR, zShared_dim=args.n_shared)
decB = DecoderB(args.wseed, num_attr=N_ATTR, zShared_dim=args.n_shared)
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


def elbo(q, pA, pB=None, lamb=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0):
    sharedA = []
    sharedB = []
    poe = []
    for i in range(N_ATTR):
        sharedA.append('sharedA' + str(i))
        sharedB.append('sharedB' + str(i))
        poe.append('poe' + str(i))

    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_own'], latents=['privateA', 'sharedA'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta1, bias=bias)

    if pB:
        reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_own'], latents=['sharedB'],
                                                                   sample_dim=0, batch_dim=1,
                                                                   beta=beta2, bias=bias)
    if q['poe'] is not None:

        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'],
                                                                         latents=['privateA', 'poe'], sample_dim=0,
                                                                         batch_dim=1,
                                                                         beta=beta1, bias=bias)
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_poe'], latents=['poe'],
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=beta2, bias=bias)

        # # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_cross'],
                                                                       latents=['privateA', 'sharedB'], sample_dim=0,
                                                                       batch_dim=1,
                                                                       beta=beta1, bias=bias)
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_cross'], latents=['sharedA'],
                                                                       sample_dim=0, batch_dim=1,
                                                                       beta=beta2, bias=bias)

        loss = (reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B) + \
               (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) + \
               (reconst_loss_crA - kl_crA) + (lamb * reconst_loss_crB - kl_crB)
    else:
        reconst_loss_poeA = reconst_loss_crA = reconst_loss_poeB = reconst_loss_crB = None
        if pB:
            loss = 3 * ((reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B))
        else:
            reconst_loss_B = None
            loss = 3 * (reconst_loss_A - kl_A)
    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
                                                                          reconst_loss_crB]


def train(data, encA, decA, encB, decB, optimizer,
          label_mask={}, fixed_imgs=None, fixed_attr=None):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    pair_cnt = 0
    encA.train()
    encB.train()
    decA.train()
    decB.train()
    N = 0
    cnt = 0

    for b, (images, attributes) in enumerate(data):
        N += 1
        if CUDA:
            images = images.cuda()
            attributes = attributes.cuda()
        optimizer.zero_grad()

        if label_mask[b]:
            cnt += 1
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

            ## poe ##
            mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')

            # decode attr
            pB, _ = decB(attributes, {'poe': 'poe', 'cross': 'sharedA', 'own': 'sharedB'}, q=q,
                         num_samples=NUM_SAMPLES)

            # decode img
            pA = decA(images, {'poe': 'poe', 'cross': 'sharedB', 'own': 'sharedA'}, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)
        else:
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)

            # decode img

            pA = decA(images, {'own': 'sharedA'}, q=q, num_samples=NUM_SAMPLES)
            loss, recA, recB = elbo(q, pA, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TRAIN)

        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()
            recA[0] = recA[0].cpu()

        epoch_elbo += loss.item()
        epoch_recA += recA[0].item()

        if recA[1] is not None:
            if CUDA:
                recB[0] = recB[0].cpu()
                for i in range(2):
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()
            epoch_recB += recB[0].item()
            epoch_rec_poeA += recA[1].item()
            epoch_rec_crA += recA[2].item()
            epoch_rec_poeB += recB[1].item()
            epoch_rec_crB += recB[2].item()
            pair_cnt += 1

        if b % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                e, b * args.batch_size, len(data.dataset),
                   100. * b * args.batch_size / len(data.dataset)))

    if pair_cnt == 0:
        pair_cnt = 1
    print('frac:', cnt / N)
    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / pair_cnt, epoch_rec_crA / pair_cnt], [epoch_recB / N,
                                                                                                   epoch_rec_poeB / pair_cnt,
                                                                                                   epoch_rec_crB / pair_cnt]


def test(data, encA, decA, encB, decB, epoch, bias):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    N = 0
    all_pred = []
    all_target = []
    for b, (images, attributes) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()

            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)


            # decode attr
            pB, pred_attr = decB(attributes, {'own': 'sharedB', 'cross': 'sharedA'}, q=q,
                                 num_samples=NUM_SAMPLES)

            # decode img
            pA = decA(images, {'own': 'sharedA', 'cross': 'sharedB'}, q=q, num_samples=NUM_SAMPLES)

            batch_elbo, _, _ = elbo(q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=bias)

            if CUDA:
                batch_elbo = batch_elbo.cpu()
                pred_attr = pred_attr.cpu()
                attributes = attributes.cpu()
            epoch_elbo += batch_elbo.item()

            pred = pred_attr.detach().numpy()
            pred = np.round(np.exp(pred))
            target = attributes.detach().numpy()
            if N == 1:
                all_pred = pred
                all_target = target
            else:
                all_pred = np.concatenate((all_pred, pred), axis=0)
                all_target = np.concatenate((all_target, target), axis=0)

    print('---------------------f1 mic------------------------')
    f1 = []
    for i in range(18):
        f1.append(
            [IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]], f1_score(all_target[:, i], all_pred[:, i], average="micro")])
    f1.sort(key=lambda f1: f1[0])
    for i in range(18):
        print(f1[i][0], f1[i][1])
    print('avg:', np.mean([elt[1] for elt in f1]))
    print('---------------------f1 mac------------------------')
    f1 = []
    for i in range(18):
        f1.append(
            [IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]], f1_score(all_target[:, i], all_pred[:, i], average="macro")])
    f1.sort(key=lambda f1: f1[0])
    for i in range(18):
        print(f1[i][0], f1[i][1])
    print('avg:', np.mean([elt[1] for elt in f1]))
    print('---------------------f1 bin------------------------')
    f1 = []
    for i in range(18):
        f1.append(
            [IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]], f1_score(all_target[:, i], all_pred[:, i], average="binary")])
    f1.sort(key=lambda f1: f1[0])
    for i in range(18):
        print(f1[i][0], f1[i][1])
    print('avg:', np.mean([elt[1] for elt in f1]))
    print('---------------------acc------------------------')
    acc = []
    for i in range(18):
        acc.append([IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]], (all_target[:, i] == all_pred[:, i]).mean()])
    acc.sort(key=lambda all_acc: all_acc[0])

    for i in range(18):
        print(acc[i][0], acc[i][1])
    print('avg:', np.mean([elt[1] for elt in acc]))
    print('-----------------------------------------------')

    return epoch_elbo / N, np.mean([elt[1] for elt in acc]), np.mean([elt[1] for elt in f1])


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

label_mask = {}

paired_idx = list(range(len(train_data)))
random.seed(args.seed)
random.shuffle(paired_idx)
paired_idx = paired_idx[:int(args.label_frac * len(train_data))]
print('paired_idx: ', paired_idx)

for b in range(len(train_data)):
    if b in paired_idx:
        label_mask[b] = True
    else:
        label_mask[b] = False

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB = train(train_data, encA, decA, encB, decB,
                                             optimizer, label_mask)
    train_end = time.time()

    val_start = time.time()
    val_elbo, val_accuracy, val_f1 = test(val_data, encA, decA, encB, decB, e, BIAS_VAL)
    val_end = time.time()

    test_start = time.time()
    test_elbo, test_accuracy, test_f1 = test(test_data, encA, decA, encB, decB, e, BIAS_TEST)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_f1=test_f1,
                           val_f1=val_f1,
                           test_acc=test_accuracy,
                           val_acc=val_accuracy,
                           test_total_loss=test_elbo,
                           val_total_loss=val_elbo,
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

    if (e + 1) % 5 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
        # util.evaluation.save_cross_celeba_cont(e, test_data, encA, decA, encB, ATTR_TO_PLOT, 64,
        #                                        args.n_shared, CUDA, MODEL_NAME)
        # util.evaluation.save_traverse_celeba_cont(e, train_data, encA, decA, CUDA, MODEL_NAME,
        #                                           fixed_idxs=[5, 10000, 22000, 30000, 45500, 50000, 60000, 70000, 75555,
        #                                                       95555],
        #                                           private=False)
        # util.evaluation.save_traverse_celeba_cont(e, test_data, encA, decA, CUDA, MODEL_NAME,
        #                                           fixed_idxs=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        #                                           private=False)
    print(
        '[Epoch %d] Train: ELBO %.4e (%ds), Val: ELBO %.4e (%ds), Test: ELBO %.4e, Accuracy %0.3f, F1-score %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start, val_elbo, val_end - val_start,
            test_elbo, test_accuracy, test_f1, test_end - test_start))

if args.ckpt_epochs == args.epochs:
    util.evaluation.save_cross_celeba_cont(args.ckpt_epochs, test_data, encA, decA, encB, ATTR_TO_PLOT, 64,
                                           args.n_shared, CUDA, MODEL_NAME)

    ### for stat
    n_batch = 10
    random.seed = 0
    fixed_idxs = random.sample(range(len(train_data.dataset)), 100 * n_batch)
    fixed_XA = [0] * 100 * n_batch
    for i, idx in enumerate(fixed_idxs):
        fixed_XA[i], _ = train_data.dataset.__getitem__(idx)[:2]
        if CUDA:
            fixed_XA[i] = fixed_XA[i].cuda()
        fixed_XA[i] = fixed_XA[i].squeeze(0)
    fixed_XA = torch.stack(fixed_XA, dim=0)

    zS_mean = 0
    # zS_ori_sum = np.zeros(zS_dim)
    for idx in range(n_batch):
        q = encA(fixed_XA[100 * idx:100 * (idx + 1)], num_samples=1)
        zS_mean += q['sharedA'].dist.loc
        # zS_std += q['sharedA'].dist.scale
    zS_mean = zS_mean.squeeze(0)
    min = zS_mean.min(dim=0)[0].detach().cpu().numpy()
    max = zS_mean.max(dim=0)[0].detach().cpu().numpy()
    ##############

    # util.evaluation.save_traverse_celeba_cont(args.ckpt_epochs, train_data, encA, decA, CUDA, MODEL_NAME,
    #                                           fixed_idxs=[5, 10000, 22000, 30000, 45500, 50000, 60000, 70000, 75555,
    #                                                       95555],
    #                                           private=False)
    util.evaluation.save_traverse_celeba_cont(args.ckpt_epochs, test_data, encA, decA, CUDA, MODEL_NAME,
                                              fixed_idxs=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
                                              private=False, min=min, max=max)
    # test_elbo, test_accuracy, test_f1 = test(test_data, encA, decA, encB, decB, 0, BIAS_TEST)
    # print('test_accuracy:', test_accuracy)
    # print('test_f1:', test_f1)

else:
    save_ckpt(args.epochs)


####
def visualize_line_metrics(self, iters, metric1, metric2):
    # prepare data to plot
    iters = torch.tensor([iters], dtype=torch.int64).detach()
    metric1 = torch.tensor([metric1])
    metric2 = torch.tensor([metric2])
    metrics = torch.stack([metric1.detach(), metric2.detach()], -1)

    VIZ.line(
        X=iters, Y=metrics, env=MODEL_NAME + '/lines',
        win=WIN_ID['metrics'], update='append',
        opts=dict(xlabel='iter', ylabel='metrics',
                  title='Disentanglement metrics',
                  legend=['metric1', 'metric2'])
    )
