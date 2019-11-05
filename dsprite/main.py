from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import visdom
from model import EncoderA, EncoderB, DecoderA, DecoderB
from dataset import Position

import sys
sys.path.append('../')
import probtorch
import util

import sys
sys.path.append('../')
from util.solver_test import Solver

#------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=0, metavar='N',
                        help='run_id')
    parser.add_argument('--dataset', type=str, default='dsprites')
    parser.add_argument('--dset_dir', type=str, default='../../data/')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=2,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_private', type=int, default=5,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=1.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=3.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=5.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')

    parser.add_argument('--ckpt_path', type=str, default='../weights/dsprites/1',
                        help='save and load path for ckpt')
    parser.add_argument( '--cuda',
      default=False, type=probtorch.util.str2bool, help='enable visdom visualization' )

    # visdom
    parser.add_argument( '--viz_on',
      default=False, type=probtorch.util.str2bool, help='enable visdom visualization' )
    parser.add_argument( '--viz_port',
      default=8002, type=int, help='visdom port number' )
    parser.add_argument( '--viz_ll_iter',
      default=1000, type=int, help='visdom line data logging iter' )
    parser.add_argument( '--viz_la_iter',
      default=5000, type=int, help='visdom line data applying iter' )
    #parser.add_argument( '--viz_ra_iter',

    args = parser.parse_args()

#------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()


if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)


# path parameters

# path parameters
MODEL_NAME = '%s-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1_%s-beta2_%s-seed%s-bs%s' %\
             (args.dataset, args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta1, args.beta2, args.seed, args.batch_size)

DATA_PATH = '../data'

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA1 = (1., args.beta1, 1.)
BETA2 = (1., args.beta2, 1.)
BIAS_TRAIN = 1.0
BIAS_TEST = 1.0
# model parameters
NUM_PIXELS = 4096
TEMP = 0.66
NUM_SAMPLES = 1

# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['recon'])
    # if self.eval_metrics:
    #     self.viz.close(env=self.name+'/lines', win=WIN_ID['metrics'])


####
def visualize_line():
    # prepare data to plot
    data = LINE_GATHER.data
    iters = torch.Tensor(data['iter'])
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])
    recons = torch.stack(
        [recon_A.detach(), recon_B.detach()], -1
    )
    VIZ.line(
        X=iters, Y=recons, env=MODEL_NAME + '/lines',
        win=WIN_ID['recon'], update='append',
        opts=dict(xlabel='iter', ylabel='recon losses',
                  title='Recon Losses', legend=['A', 'B'])
    )

if args.viz_on:
    WIN_ID = dict(
        recon='win_recon'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'iter', 'recon_A', 'recon_B'
    )

    viz_port = args.viz_port  # port number, eg, 8097
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()
    viz_ll_iter = args.viz_ll_iter
    viz_la_iter = args.viz_la_iter


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())

encA = EncoderA(num_pixels=4096, num_hidden=512, zPrivate_dim=args.n_private, zShared_dim=args.n_shared)
decA = DecoderA(num_pixels=4096, num_hidden=512, zPrivate_dim=args.n_private, zShared_dim=args.n_shared)
encB = EncoderB(num_pixels=4096, num_hidden=512, zPrivate_dim=args.n_private, zShared_dim=args.n_shared)
decB = DecoderB(num_pixels=4096, num_hidden=512, zPrivate_dim=args.n_private, zShared_dim=args.n_shared)
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


def elbo(iter, q, pA, pB, lamb=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0):
    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['imagesA_sharedA'], latents=['privateA', 'sharedA'], sample_dim=0, batch_dim=1,
                                        beta=beta1, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['imagesB_sharedB'], latents=['privateB', 'sharedB'], sample_dim=0, batch_dim=1,
                                        beta=beta2, bias=bias)
    if q['poe'] is not None:
        # by POE
        # 기대하는바:sharedA가 sharedB를 따르길. 즉 sharedA에만 digit정보가 있으며, 그 permutataion이 GT에서처럼 identity이기를
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['imagesA_poe'], latents=['privateA', 'poe'], sample_dim=0, batch_dim=1,
                                                    beta=beta1, bias=bias)
        # 의미 없음. rec 은 항상 0. 인풋이 항상 GT고 poe결과도 GT를 따라갈 확률이 크기 때문(학습 초반엔 A가 unif이라서, 학습 될수록 A가 B label을 잘 따를테니)
        #loss 값 변화 자체로는 의미 없지만, 이 일정한 loss(GT)가 나오도록하는 sharedA의 back pg에는 의미가짐
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['imagesB_poe'], latents=['privateB', 'poe'], sample_dim=0, batch_dim=1,
                                                    beta=beta2, bias=bias)

        # # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['imagesA_sharedB'], latents=['privateA', 'sharedB'], sample_dim=0, batch_dim=1,
                                                    beta=beta1, bias=bias)
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['imagesB_sharedA'], latents=['privateA', 'sharedB'], sample_dim=0, batch_dim=1,
                                                    beta=beta2, bias=bias)

        loss = (reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B) + \
               (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) + \
               (reconst_loss_crA - kl_crA) + (lamb * reconst_loss_crB - kl_crB)
    else:
        loss = 3*((reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B))

    if args.viz_on and (iter % args.viz_ll_iter == 0):
        LINE_GATHER.insert(iter=iter,
                           recon_A=reconst_loss_A.item(),
                           recon_B=reconst_loss_B.item()
                           )
    return loss

def train(data, encA, decA, encB, decB, optimizer,
          label_mask={}, fixed_imgs=None, fixed_labels=None):
    epoch_elbo = 0.0
    encA.train()
    encA.train()
    decA.train()
    decA.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)
    for b, (imagesA, imagesB, _) in enumerate(data):
        N += args.batch_size
        imagesA = imagesA.view(-1, NUM_PIXELS)
        imagesB = imagesB.view(-1, NUM_PIXELS)
        if CUDA:
            imagesA = imagesA.cuda()
            imagesB = imagesB.cuda()
        optimizer.zero_grad()
        if b not in label_mask:
            label_mask[b] = (random.random() < args.label_frac)
        if (label_mask[b] and args.label_frac == args.sup_frac):
            # encode
            q = encA(imagesA, num_samples=NUM_SAMPLES)
            q = encB(imagesB, num_samples=NUM_SAMPLES, q=q)
            ## poe ##
            mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')

            # decode
            pA = decA(imagesA, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(imagesB, {'sharedA': q['sharedA'], 'sharedB': q['sharedB'], 'poe': q['poe']}, q=q,
                      num_samples=NUM_SAMPLES)
            for param in encB.parameters():
                param.requires_grad = True
            for param in decB.parameters():
                param.requires_grad = True
            # loss
            loss = -elbo(b, q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2,  bias=BIAS_TRAIN)
        else:
            # encode
            q = encA(imagesA, num_samples=NUM_SAMPLES)
            q = encB(imagesB, num_samples=NUM_SAMPLES, q=q)

            # decode
            pA = decA(imagesA, {'sharedA': q['sharedA']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(imagesB, {'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)
            for param in encB.parameters():
                param.requires_grad = False
            for param in decB.parameters():
                param.requires_grad = False
            # loss
            loss = -elbo(b, q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2,  bias=BIAS_TRAIN)

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
    N = 0
    for b, (imagesA, imagesB, _) in enumerate(data):
        if imagesA.size()[0] == args.batch_size:
            N += args.batch_size
            imagesA = imagesA.view(-1, NUM_PIXELS)
            imagesB = imagesB.view(-1, NUM_PIXELS)
            if CUDA:
                imagesA = imagesA.cuda()
                imagesB = imagesB.cuda()
            # encode
            q = encA(imagesA, num_samples=NUM_SAMPLES)
            q = encB(imagesB, num_samples=NUM_SAMPLES, q=q)

            # decode
            pA = decA(imagesA, {'sharedA': q['sharedA']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(imagesB, {'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)

            batch_elbo = elbo(b, q, pA, pB, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TEST)

            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            # epoch_correct += pB['labels_sharedA'].loss.sum().item()
    if epoch % 2 ==0 or epoch + 1 == args.epochs:
        metric1A, _ = solverA.eval_disentangle_metric1()
        metric2A, _ = solverA.eval_disentangle_metric2()
        metric1B, _ = solverB.eval_disentangle_metric1()
        metric2B, _ = solverB.eval_disentangle_metric2()

    else:
        metric1A, metric2A, metric1B, metric2B = -1, -1, -1, -1
    return epoch_elbo / N, metric1A, metric2A, metric1B, metric2B


def get_paired_data(paired_cnt, seed):
    data = torch.utils.data.DataLoader(Position(), batch_size=args.batch_size, shuffle=True)
    tr_labels = data.dataset.targets

    cnt = int(paired_cnt / 10)
    assert cnt == paired_cnt / 10

    label_idx = {}
    for i in range(10):
        label_idx.update({i:[]})
    for idx in  range(len(tr_labels)):
        label = int(tr_labels[idx].data.detach().cpu().numpy())
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
        img, label, _ = data.dataset.__getitem__(idx)
        imgs.append(img)
        labels.append(torch.tensor(label))
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)

    # train_kwargs = {'data_tensorA': imgs, 'data_tensorB': labels}
    # dataset=util.dataset.CustomDataset(**train_kwargs)
    #
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
    #                         num_workers=0, pin_memory=True, drop_last=True)
    # return dataloader
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

# latents={'private': 'privateA', 'shared':'sharedA'}
# metric1, C = util.evaluation.eval_disentangle_metric1(CUDA, encA, args.n_private,
#                                                       args.n_shared, latents, num_pixels=4096, dataset='dsprites')
# metric2, C = util.evaluation.eval_disentangle_metric2(CUDA, encA, args.n_private,
#                                                       args.n_shared, latents, num_pixels=4096, dataset='dsprites')

#
# latents={'private': 'privateB', 'shared':'sharedB'}
# metric1, C = util.evaluation.eval_disentangle_metric1(CUDA, encB, args.n_private,
#                                                       args.n_shared, latents, num_pixels=4096, dataset='3dfaces')
# metric2, C = util.evaluation.eval_disentangle_metric2(CUDA, encB, args.n_private,
#                                                       args.n_shared, latents, num_pixels=4096, dataset='3dfaces')

args.model_name = MODEL_NAME
args.num_pixels = NUM_PIXELS

args.dataset='dsprites'
args.enc = encA
args.latents={'private': 'privateA', 'shared':'sharedA'}
solverA = Solver(args)

args.dataset='3dfaces'
args.enc = encB
args.latents={'private': 'privateB', 'shared':'sharedB'}
solverB = Solver(args)


mask = {}
fixed_imgs=None
fixed_labels=None

train_data = torch.utils.data.DataLoader(Position(), batch_size=args.batch_size, shuffle=True)
test_data = train_data

if args.label_frac > 1:
    fixed_imgs, fixed_labels = get_paired_data(args.label_frac, args.seed)


for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()

    train_elbo, mask = train(train_data, encA, decA, encB, decB,
                             optimizer, mask, fixed_imgs=fixed_imgs, fixed_labels=fixed_labels)

    train_end = time.time()
    test_start = time.time()
    test_elbo, metric1A, metric2A, metric1B, metric2B = test(test_data, encA, decA, encB, decB, e)
    test_end = time.time()
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Metric1A %0.3f, Metric2A %0.3f, Metric1B %0.3f, Metric2B %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start,
            test_elbo, metric1A, metric2A, metric1B, metric2B, test_end - test_start))


    # (visdom) visualize line stats (then flush out)
    if args.viz_on and (e % args.viz_la_iter == 0):
        visualize_line()
        LINE_GATHER.flush()


if not os.path.isdir(args.ckpt_path):
    os.mkdir(args.ckpt_path)
torch.save(encA.state_dict(),
           '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))
torch.save(decA.state_dict(),
           '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))
torch.save(encB.state_dict(),
           '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))
torch.save(decB.state_dict(),
           '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))

# util.evaluation.mutual_info(test_data, encA, CUDA, flatten_pixel=NUM_PIXELS)
# util.evaluation.save_traverse(args.epochs, test_data, encA, decA, CUDA, fixed_idxs=[3, 2, 1, 18, 4, 15, 11, 17, 61, 99], output_dir_trvsl=MODEL_NAME, flatten_pixel=NUM_PIXELS)


####
def visualize_line_metrics(iters, metric1, metric2):
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
