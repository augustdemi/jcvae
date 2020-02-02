from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import visdom
import numpy as np

from model import EncoderA
from datasets import datasets
from sklearn.metrics import f1_score

import sys

sys.path.append('../')
import probtorch
import util
from torch.nn import functional as F

# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1, metavar='N',
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
    parser.add_argument('--epochs', type=int, default=85, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=100.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=5.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/celeba_clf/',
                        help='save and load path for ckpt')

    parser.add_argument('--attr', type=str, default='Male',
                        help='save cross gen img from attr')

    # visdom
    parser.add_argument('--viz_on',
                        default=False, type=probtorch.util.str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port',
                        default=8002, type=int, help='visdom port number')

    args = parser.parse_args()

# ------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

print('>>>CUDA:', CUDA)

# path parameters
MODEL_NAME = 'celeba_clf-run_id%d-priv%02ddim-shared%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s-lr%s' % (
    args.run_id, args.n_private, args.n_shared, args.label_frac, args.sup_frac, args.lambda_text, args.beta1,
    args.beta2, args.seed,
    args.batch_size, args.wseed, args.lr)
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
ATTR_TO_PLOT = ['Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Blond_Hair', 'Eyeglasses', 'Bangs', 'off',
                'Black_Hair', 'Wavy_Hair']


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['f1'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


def visualize_line():
    data = LINE_GATHER.data
    total_loss = torch.Tensor(data['total_loss'])

    epoch = torch.Tensor(data['epoch'])
    test_acc = torch.Tensor(data['test_acc'])
    test_f1 = torch.Tensor(data['test_f1'])

    total_losses = torch.tensor(np.stack([total_loss], -1))
    acc = torch.tensor(np.stack([test_acc], -1))
    f1 = torch.tensor(np.stack([test_f1], -1))

    VIZ.line(
        X=epoch, Y=acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Accuracy', legend=['test_acc'])
    )

    VIZ.line(
        X=epoch, Y=f1, env=MODEL_NAME + '/lines',
        win=WIN_ID['f1'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='F1 score', legend=['test_f1'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        acc='win_acc', total_losses='win_total_losses', f1='win_f1'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch',
        'total_loss', 'test_acc', 'test_f1'
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

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, n_attr=N_ATTR)
if CUDA:
    encA.cuda()
    cuda_tensors(encA)

optimizer = torch.optim.Adam(
    list(encA.parameters()),
    lr=args.lr)


def train(data, encA, optimizer):
    encA.train()
    N = 0
    total_loss = 0
    for b, (images, attributes) in enumerate(data):
        N += 1
        optimizer.zero_grad()
        if CUDA:
            images = images.cuda()
            attributes = attributes.cuda()

        # encode
        pred_attr = encA(images, num_samples=NUM_SAMPLES)

        loss = F.binary_cross_entropy_with_logits(pred_attr, attributes, reduction='none').sum()
        if CUDA:
            loss = loss.cuda()
        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()

        total_loss += loss

        if b % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                e, b * args.batch_size, len(data.dataset),
                   100. * b * args.batch_size / len(data.dataset)))
    return total_loss


def test(data, encA):
    encA.eval()
    epoch_elbo = 0.0
    epoch_acc = 0
    epoch_f1 = 0
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
            pred_attr = encA(images, num_samples=NUM_SAMPLES)

            if CUDA:
                pred_attr = pred_attr.cpu()
                attributes = attributes.cpu()

            pred = pred_attr.detach().numpy()
            pred = np.round(np.exp(pred))
            target = attributes.detach().numpy()
            epoch_acc += (pred == target).mean()
            epoch_f1 += f1_score(target, pred, average="samples")

            if N == 1:
                all_pred = pred
                all_target = target
            else:
                all_pred = np.concatenate((all_pred, pred), axis=0)
                all_target = np.concatenate((all_target, target), axis=0)

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

    return epoch_acc / N, epoch_f1 / N


def save_ckpt(e):
    if not os.path.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    torch.save(encA.state_dict(),
               '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


if args.ckpt_epochs > 0:
    if CUDA:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
    else:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))

mask = {}
fixed_imgs = None
fixed_attr = None

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    total_loss = train(train_data, encA, optimizer)
    train_end = time.time()

    test_start = time.time()
    test_accuracy, test_f1 = test(test_data, encA)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_f1=test_f1,
                           test_acc=test_accuracy,
                           total_loss=total_loss
                           )
        visualize_line()
        LINE_GATHER.flush()
    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)

    print(
        '[Epoch %d] Train: ELBO %.4e (%ds), Test: Accuracy %0.3f, F1-score %0.3f (%ds)' % (
            e, total_loss, train_end - train_start, test_accuracy, test_f1, test_end - test_start))

if args.ckpt_epochs == args.epochs:

    util.evaluation.save_cross_celeba(args.ckpt_epochs, test_data, encA, decA, encB, ATTR_TO_PLOT, 64, args.n_shared,
                                      CUDA, MODEL_NAME)

    # util.evaluation.save_traverse_celeba(args.ckpt_epochs, train_data, encA, decA, args.n_shared, CUDA, MODEL_NAME,
    #                                      fixed_idxs=[5, 10000, 22000, 30000, 45500, 50000, 60000, 70000, 75555, 95555],
    #                                      private=False)

    util.evaluation.save_traverse_celeba(args.ckpt_epochs, test_data, encA, decA, args.n_shared, CUDA, MODEL_NAME,
                                         fixed_idxs=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
                                         private=False)


    # test_elbo, test_accuracy, test_f1 = test(test_data, encA, decA, encB, decB, 0, BIAS_TEST)
    # print('test_accuracy:', test_accuracy)
    # print('test_f1:', test_f1)
    #

else:
    save_ckpt(args.epochs)

