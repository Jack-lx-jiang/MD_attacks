import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from attack import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--gpus', nargs='+', type=int, required=True, help='The gpus to use')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8. / 255., type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=40, type=int,
                    help='perturb number of steps')
parser.add_argument('--first-step-size', default=16. / 255., type=float,
                    help='perturb step size for first stage')
parser.add_argument('--step-size', default=2. / 255., type=float,
                    help='perturb step size for second stage')
parser.add_argument('--random',
                    type=int,
                    default=1,
                    help='number of random initialization for PGD')
parser.add_argument('--adv-save-path', default='', help='the path to save adv')

# model setting
parser.add_argument('--model-path',
                    default='',
                    help='model for white-box attack evaluation')

# attack type
parser.add_argument('--pgd', action='store_true', default=False)
parser.add_argument('--md', action='store_true', default=False)
parser.add_argument('--mdmt', action='store_true', default=False)

args = parser.parse_args()

# settings
gpus = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def eval_adv_test_whitebox(model, device, test_loader, vmin, vmax, attack_args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_correct_total = 0
    natural_correct_total = 0

    adv_list = []
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)

        if attack_args.md:
            nat_correct, adv_correct, X_adv = MD_attack(model, X, y, epsilon=attack_args.epsilon,
                                                        num_steps=attack_args.num_steps,
                                                        step_size=attack_args.step_size,
                                                        num_random_starts=attack_args.random,
                                                        v_min=vmin, v_max=vmax,
                                                        change_point=attack_args.num_steps / 2,
                                                        first_step_size=attack_args.first_step_size)
        elif attack_args.mdmt:
            nat_correct, adv_correct, X_adv = MDMT_attack(model, X, y, epsilon=attack_args.epsilon,
                                                          num_steps=attack_args.num_steps,
                                                          step_size=attack_args.step_size,
                                                          v_min=vmin, v_max=vmax,
                                                          change_point=attack_args.num_steps / 2,
                                                          first_step_size=attack_args.first_step_size)
        else:
            nat_correct, adv_correct, X_adv = PGD_attack(model, X, y, epsilon=attack_args.epsilon,
                                                         num_steps=attack_args.num_steps,
                                                         step_size=attack_args.step_size,
                                                         num_random_starts=attack_args.random,
                                                         v_min=vmin, v_max=vmax)

        pertub = X_adv - X.detach().cpu().clone().numpy()
        valid = (pertub <= attack_args.epsilon * (vmax - vmin) + 1e-6) & (
                pertub >= -attack_args.epsilon * (vmax - vmin) - 1e-6)
        assert np.all(valid), 'perturb outrange!'
        adv_list.append(X_adv)
        num_adv_correct = adv_correct.float().sum().item()
        print('adv correct (white-box): ', num_adv_correct)
        natural_correct_total += nat_correct.float().sum().item()
        robust_correct_total += num_adv_correct

    print('natural_correct_total: ', natural_correct_total)
    print('robust_correct_total: ', robust_correct_total)
    adv_dataset = np.concatenate(adv_list, axis=0)
    return adv_dataset


def main():
    # set up data loader
    vmax = 1.0
    vmin = 0.
    transform_test = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    '''
    Please load your own model here.
    Be careful of the input range (change the vmax, vmin and transformer accordingly).
    '''
    assert False, 'Please load your own models.'
    model = WideResNet(depth=28,
                       num_classes=10,
                       widen_factor=10).to(device)
    # model.load_state_dict(torch.load(model_path))
    model.eval()

    adv_dataset = eval_adv_test_whitebox(model, device, test_loader, vmin, vmax, args)
    if args.adv_save_path != '':
        np.save(args.adv_save_path, adv_dataset)


if __name__ == '__main__':
    main()
