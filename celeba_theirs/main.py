from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
import random

from datasets import datasets
from model import Encoder, Decoder
from sklearn.metrics import f1_score
import visdom
import sys

sys.path.append('../')
import probtorch
import util

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=10,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_private', type=int, default=50,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=-1.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta', type=float, default=-1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')

    parser.add_argument('--ckpt_path', type=str, default='../weights/svhn_theirs',
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
MODEL_NAME = 'celeba_theirs-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta%s-seed%s' % (
    args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta, args.seed)
DATA_PATH = '../data'

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
NUM_SAMPLES = 1
N_ATTR = 18
ATTR_TO_PLOT = ['Receding_Hairline', 'Bushy_Eyebrows', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling',
                'Blond_Hair', 'Eyeglasses', 'Bangs', 'off',
                'Black_Hair', 'Wavy_Hair']

if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA = (1., args.beta, 1.)

# model parameters
NUM_PIXELS = None
TEMP = 0.66
NUM_SAMPLES = 1

if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['f1'])


def visualize_line():
    data = LINE_GATHER.data

    epoch = torch.Tensor(data['epoch'])
    f1 = torch.Tensor(data['test_f1'])

    VIZ.line(
        X=epoch, Y=f1, env=MODEL_NAME + '/lines',
        win=WIN_ID['f1'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='F1 score', legend=['test_f1'])
    )


if args.viz_on:
    WIN_ID = dict(
        f1='win_f1'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'test_f1'
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
                                         shuffle=True)

test_data = torch.utils.data.DataLoader(datasets(partition='test', data_dir='../../data/celeba2',
                                                 image_transform=preprocess_data), batch_size=args.batch_size,
                                        shuffle=False)
val_data = torch.utils.data.DataLoader(datasets(partition='val', data_dir='../../data/celeba2',
                                                image_transform=preprocess_data), batch_size=args.batch_size,
                                       shuffle=False)

print('>>> data loaded')
print('train: ', len(train_data.dataset))
print('test: ', len(test_data.dataset))


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


enc = Encoder(num_attr=18, num_style=args.n_private)
dec = Decoder(num_attr=18, num_style=args.n_private)
if CUDA:
    enc.cuda()
    dec.cuda()
    cuda_tensors(enc)
    cuda_tensors(dec)

optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                             lr=args.lr)


def elbo(q, p, alpha=0.1):
    if NUM_SAMPLES is None:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
    else:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)


FIXED = []


def train(data, enc, dec, optimizer,
          label_mask={}, label_fraction=args.label_frac, fixed_imgs=None, fixed_attr=None):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    for b, (images, attributes) in enumerate(data):
        if images.size()[0] == args.batch_size:
            if args.label_frac > 1 and random.random() < args.sup_frac:
                # print(b)
                N += 1
                shuffled_idx = list(range(int(args.label_frac)))
                random.shuffle(shuffled_idx)
                shuffled_idx = shuffled_idx[:args.batch_size]
                # print(shuffled_idx[:10])
                images = fixed_imgs[shuffled_idx]
                attributes = fixed_attr[shuffled_idx]
                optimizer.zero_grad()
                if CUDA:
                    images = images.cuda()
                    attributes = attributes.cuda()

                optimizer.zero_grad()
                q = enc(images, attributes, num_samples=NUM_SAMPLES)
                p = dec(images, q, num_samples=NUM_SAMPLES)
                loss = -elbo(q, p)
            else:
                N += args.batch_size
                if CUDA:
                    images = images.cuda()
                    attributes = attributes.cuda()
                optimizer.zero_grad()
                if b not in label_mask:
                    label_mask[b] = (random.random() < label_fraction)
                if (label_mask[b] and args.label_frac == args.sup_frac):
                    q = enc(images, attributes, num_samples=NUM_SAMPLES)
                else:
                    q = enc(images, num_samples=NUM_SAMPLES)
                p = dec(images, q, num_samples=NUM_SAMPLES)
                loss = -elbo(q, p)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_elbo -= loss.item()
    return epoch_elbo / N, label_mask


import numpy as np


def test(data, enc, dec, infer=True):
    enc.eval()
    dec.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0

    all_pred = []
    all_target = []
    for b, (images, attributes) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += args.batch_size
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES)
            batch_elbo = elbo(q, p)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            if infer:
                pred_attr = []
                for i in range(18):
                    log_p = p.log_joint(0, 1)
                    log_q = q.log_joint(0, 1)
                    log_w = log_p - log_q
                    w = torch.nn.functional.softmax(log_w, 0)

                    y_samples = q['attr' + str(i)].value
                    y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
                    _, y_pred = y_expect.max(-1)
                    pred_attr.append(y_pred.unsqueeze(1))
                pred_attr = torch.cat(pred_attr, -1)

            else:
                pred_attr = []
                for i in range(18):
                    _, y_pred = q['attr' + str(i)].value.max(-1)
                    pred_attr.append(y_pred.unsqueeze(1))
                pred_attr = torch.cat(pred_attr, -1)

            if CUDA:
                pred_attr = pred_attr.cpu()
                attributes = attributes.cpu()
                

            if N == args.batch_size:
                all_pred = pred_attr
                all_target = attributes
            else:
                all_pred = np.concatenate((all_pred, pred_attr), axis=0)
                all_target = np.concatenate((all_target, attributes), axis=0)

    print('---------------------f1------------------------')
    f1 = []
    for i in range(18):
        f1.append(f1_score(all_target[:, i], all_pred[:, i], average="binary"))

    f1 = list(enumerate(f1))
    f1.sort(key=lambda f1: f1[1])

    for i in range(18):
        print(IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[f1[i][0]]], f1[i][1])
    print('---------------------acc------------------------')

    all_acc = []
    for i in range(18):
        acc = (all_target[:, i] == all_pred[:, i]).mean()
        all_acc.append(acc)

    all_acc = list(enumerate(all_acc))
    all_acc.sort(key=lambda all_acc: all_acc[1])

    for i in range(18):
        print(IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[all_acc[i][0]]], all_acc[i][1])
    print('-----------------------------------------------')

    print("=======================================")
    avg_f1 = np.array(f1)[:, 1].mean()
    avg_acc = np.array(all_acc)[:, 1].mean()
    print('avg_f1: ', avg_f1)
    print('avg_acc: ', avg_acc)
    print("=======================================")
    return epoch_elbo / N, avg_f1


def get_paired_data(paired_cnt, seed):
    data = torch.utils.data.DataLoader(datasets(partition='train', data_dir='../../data/celeba2',
                                                image_transform=preprocess_data), batch_size=args.batch_size,
                                       shuffle=False)
    random.seed(seed)
    total_random_idx = random.sample(range(len(data.dataset)), int(paired_cnt))

    imgs = []
    attrs = []
    for idx in total_random_idx:
        img, attr = data.dataset.__getitem__(idx)
        imgs.append(img)
        attrs.append(torch.tensor(attr))
    imgs = torch.stack(imgs, dim=0)
    attrs = torch.stack(attrs, dim=0)
    return imgs, attrs


import time

if args.ckpt_epochs > 0:
    if CUDA:
        enc.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        dec.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
    else:
        enc.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                       map_location=torch.device('cpu')))
        dec.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                       map_location=torch.device('cpu')))
        # enc.load_state_dict(torch.load(args.ckpt_path + '/mnist-semisupervised-50dim-0.0+5a2c637-1.2.0-enc.rar',
        #                                 map_location=torch.device('cpu')))
        # dec.load_state_dict(torch.load(args.ckpt_path+'/mnist-semisupervised-50dim-0.0+5a2c637-1.2.0-dec.rar',
        #                                 map_location=torch.device('cpu')))

mask = {}
fixed_imgs = None
fixed_attr = None
if args.label_frac > 1:
    fixed_imgs, fixed_attr = get_paired_data(args.label_frac, args.seed)

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, mask = train(train_data, enc, dec,
                             optimizer, mask, fixed_imgs=fixed_imgs, fixed_attr=fixed_attr)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, enc, dec)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_f1=test_accuracy
                           )
        visualize_line()
        LINE_GATHER.flush()

    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
        e, train_elbo, train_end - train_start,
        test_elbo, test_accuracy, test_end - test_start))

if not os.path.isdir(args.ckpt_path):
    os.mkdir(args.ckpt_path)
torch.save(enc.state_dict(),
           '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))
torch.save(dec.state_dict(),
           '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))

if args.ckpt_epochs == args.epochs:
    # util.evaluation.mutual_info(test_data, enc, CUDA, flatten_pixel=NUM_PIXELS, baseline=True)
    # util.evaluation.svhn_base_latent(test_data, enc, 1000)

    util.evaluation.cross_acc_svhn_baseline(test_data, enc, dec, 1000, args.n_shared,
                                            CUDA)

    util.evaluation.save_traverse_base(args.epochs, test_data, enc, dec, CUDA,
                                       fixed_idxs=[3, 2, 1, 32, 4, 23, 21, 36, 61, 99], output_dir_trvsl=MODEL_NAME,
                                       flatten_pixel=NUM_PIXELS)
    #
    # util.evaluation.save_cross_mnist_base(e, dec, enc, 64,
    #                                           CUDA, MODEL_NAME)
