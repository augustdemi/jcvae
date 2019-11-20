from torchvision import datasets, transforms

import time
import random
import torch
import os

from model import EncoderA, EncoderB, DecoderA, DecoderB

import sys
sys.path.append('../')
import probtorch
import util



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
    parser.add_argument('--beta', type=float, default=3.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')

    parser.add_argument('--ckpt_path', type=str, default='../weights/svhn',
                        help='save and load path for ckpt')

    args = parser.parse_args()

#------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'svhn-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta%s-seed%s-lr%s-bs%s' % (args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta, args.seed, args.lr, args.batch_size)
DATA_PATH = '../data'

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA = (1., args.beta, 1.)
# BIAS_TRAIN = 1.0
# BIAS_TEST = 1.0
# model parameters
# NUM_PIXELS = 3*32*32


NUM_PIXELS = None
TEMP = 0.66

NUM_SAMPLES = 1
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

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



def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())

encA = EncoderA(zPrivate_dim=args.n_private)
decA = DecoderA(zPrivate_dim=args.n_private)
encB = EncoderB()
decB = DecoderB()
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


def elbo(iter, q, pA, pB, lamb=1.0, beta=(1.0, 1.0, 1.0), bias=1.0):
    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedA'], latents=['privateA', 'sharedA'], sample_dim=0, batch_dim=1,
                                        beta=beta, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_sharedB'], latents=['sharedB'],
                                                                   sample_dim=0, batch_dim=1,
                                                               beta=beta, bias=bias)

    if q['poe'] is not None:
        # by POE
        # 기대하는바:sharedA가 sharedB를 따르길. 즉 sharedA에만 digit정보가 있으며, 그 permutataion이 GT에서처럼 identity이기를
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'], latents=['privateA', 'poe'], sample_dim=0, batch_dim=1,
                                                    beta=beta, bias=bias)
        # 의미 없음. rec 은 항상 0. 인풋이 항상 GT고 poe결과도 GT를 따라갈 확률이 크기 때문(학습 초반엔 A가 unif이라서, 학습 될수록 A가 B label을 잘 따를테니)
        #loss 값 변화 자체로는 의미 없지만, 이 일정한 loss(GT)가 나오도록하는 sharedA의 back pg에는 의미가짐
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_poe'], latents=['poe'], sample_dim=0, batch_dim=1,
                                                    beta=beta, bias=bias)

        # # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedB'], latents=['privateA', 'sharedB'], sample_dim=0, batch_dim=1,
                                                    beta=beta, bias=bias)
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_sharedA'], latents=['sharedA'], sample_dim=0, batch_dim=1,
                                                    beta=beta, bias=bias)

        # loss = (reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B) + \
        #        (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) \
        #loss = (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB)
        loss = (reconst_loss_A - kl_A) + lamb * (reconst_loss_B - kl_B) + \
               (reconst_loss_poeA - kl_poeA) + lamb * (reconst_loss_poeB - kl_poeB) + \
               (reconst_loss_crA - kl_crA) + lamb * (reconst_loss_crB - kl_crB)

        # if iter % 100 == 0:
        #     print('=========================================')
        #     print('reconst_loss_poeA: ', reconst_loss_poeA)
        #     print('kl_poeA: ', kl_poeA)
        #     print('-----------------------------------------')
        #     print('reconst_loss_poeB: ', reconst_loss_poeB)
        #     print('kl_poeB: ', kl_poeB)
        #     print('-----------------------------------------')
        #     print('reconst_loss_crA: ', reconst_loss_crA)
        #     print('kl_crA: ', kl_crA)
        #     print('-----------------------------------------')
        #     print('reconst_loss_crB: ', reconst_loss_crB)
        #     print('kl_crB: ', kl_crB)
        #     print('-----------------------------------------')
    else:
        loss = 3*((reconst_loss_A - kl_A) + (lamb * reconst_loss_B - kl_B))

    # if iter % 100 == 0:
    #     print('reconst_loss_A: ', reconst_loss_A)
    #     print('kl_A: ', kl_A)
    #     print('-----------------------------------------')
    #     print('reconst_loss_B: ', reconst_loss_B)
    #     print('kl_B: ', kl_B)
    #     print('-----------------------------------------')
    #     print('loss: ', loss)
    #     print(iter)
    #     print('=========================================')

    return loss

def train(data, encA, decA, encB, decB, optimizer,
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
                loss = -elbo(b, q, pA, pB, lamb=args.lambda_text, beta=BETA, bias=BIAS_TRAIN)
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
                    loss = -elbo(b, q, pA, pB, lamb=args.lambda_text, beta=BETA, bias=BIAS_TRAIN)
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
                    loss = -elbo(b, q, pA, pB, lamb=args.lambda_text, beta=BETA, bias=BIAS_TRAIN)

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

            batch_elbo = elbo(b, q, pA, pB, lamb=args.lambda_text, beta=BETA, bias=BIAS_TEST)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            epoch_correct += pB['labels_sharedA'].loss.sum().item()
            # print('--------------------------------iter ', b, '---------------------------------------')
            # print('sharedA')
            # cnt = [0] * 10
            # for elt in q['sharedA'].value.argmax(dim=2)[0]:
            #     cnt[elt] += 1
            # print(cnt)
            #
            # print('sharedB')
            # # print(q['sharedB'].value.argmax(dim=2)[0][:20])
            # cnt = [0] * 10
            # for elt in q['sharedB'].value.argmax(dim=2)[0]:
            #     cnt[elt] += 1
            # print(cnt)
            #
            # print('labels')
            # cnt = [0] * 10
            # for elt in labels:
            #     cnt[elt] += 1
            # print(cnt)
            #
            # print('loss')
            # print(pB['labels_sharedA'].loss.sum().item())
            # print(1 + pB['labels_sharedA'].loss.sum().item() / 100)


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
                             optimizer, mask, fixed_imgs=fixed_imgs, fixed_labels=fixed_labels)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB, e)
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
