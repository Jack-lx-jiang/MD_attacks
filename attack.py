import time
import sys
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn


def one_hot_tensor(y_batch_tensor, num_classes):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


def adv_check_and_update(X_cur, logits, y, not_correct, X_adv):
    adv_pred = logits.max(1)[1]
    nc = (adv_pred != y.data)
    not_correct += nc.long()
    X_adv[nc] = X_cur[nc]


def MD_attack(model,
              X,
              y,
              epsilon=8. / 255.,
              num_steps=40,
              step_size=2. / 255.,
              num_random_starts=1,
              v_min=0.,
              v_max=1.,
              change_point=20,
              first_step_size=16./255.,
              num_classes=10):
    epsilon = epsilon * (v_max - v_min)
    step_size = step_size * (v_max - v_min)
    first_step_size = first_step_size * (v_max - v_min)

    assert num_steps >= change_point, 'step number must be greater than change point {}'.format(change_point)
    nat_logits = model(X)
    nat_pred = nat_logits.max(dim=1)[1]
    nat_correct = (nat_pred == y).squeeze()

    y_gt = one_hot_tensor(y, num_classes)

    not_correct = torch.zeros_like(y)
    X_adv = X.detach().clone()
    for _ in range(max(num_random_starts, 1)):
        for r in range(2):
            X_pgd = Variable(X.data, requires_grad=True)
            if num_random_starts:
                random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
                X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

            for i in range(num_steps):
                with torch.enable_grad():
                    logits = model(X_pgd)
                    z_max, max_idx = torch.max(logits * (1 - y_gt) - y_gt * 100000, dim=1)
                    z_y, _ = torch.max(logits * y_gt - (1 - y_gt) * 100000, dim=1)

                    if i<1:
                        loss_per_sample = z_y
                    elif i < change_point:
                        loss_per_sample = z_max if r else -z_y
                    else:
                        loss_per_sample = z_max - z_y
                    loss = torch.mean(loss_per_sample)

                    adv_check_and_update(X_pgd, logits, y, not_correct, X_adv)

                loss.backward()

                if i < 1:
                    eta = 2 * epsilon * X_pgd.grad.data.sign()
                elif i < change_point:
                    eta = first_step_size * X_pgd.grad.data.sign()
                else:
                    eta = step_size * X_pgd.grad.data.sign()
                X_pgd = X_pgd.detach() + eta.detach()
                X_pgd = torch.min(torch.max(X_pgd, X - epsilon), X + epsilon)
                X_pgd = Variable(torch.clamp(X_pgd, v_min, v_max), requires_grad=True)
            adv_check_and_update(X_pgd, model(X_pgd), y, not_correct, X_adv)

    adv_correct = (not_correct == 0).squeeze()
    return nat_correct, adv_correct, X_adv.detach().cpu().numpy()


def MDMT_attack(model,
                X,
                y,
                epsilon=8. / 255.,
                num_steps=40,
                step_size=2. / 255.,
                v_min=0.,
                v_max=1.,
                change_point=20,
                first_step_size = 16. / 255.,
                num_classes=10):
    epsilon = epsilon * (v_max - v_min)
    step_size = step_size * (v_max - v_min)
    first_step_size = first_step_size * (v_max - v_min)

    assert num_steps >= change_point, 'step number must be greater than change point {}'.format(change_point)
    nat_logits = model(X)
    nat_pred = nat_logits.max(dim=1)[1]
    nat_correct = (nat_pred == y).squeeze()

    not_correct = torch.zeros_like(y)
    X_adv = X.detach().clone()
    for t in range(num_classes):
        targets = torch.zeros_like(y)
        targets += t

        y_tg = one_hot_tensor(targets, num_classes)
        y_gt = one_hot_tensor(y, num_classes)

        for r in range(2):
            X_pgd = Variable(X.data, requires_grad=True)
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

            for i in range(num_steps):
                with torch.enable_grad():
                    logits = model(X_pgd)

                    z_t, _ = torch.max(y_tg * logits - (1 - y_tg) * 10000, dim=1)
                    z_y, _ = torch.max(y_gt * logits - (1 - y_gt) * 10000, dim=1)

                    if i<1:
                        loss = torch.mean(z_y)
                    elif i < change_point:
                        loss = torch.mean(z_t) if r else torch.mean(- z_y)
                    else:
                        loss = torch.mean(z_t - z_y)

                    adv_check_and_update(X_pgd, logits, y, not_correct, X_adv)
                loss.backward()

                if i < 1:
                    eta = 2 * epsilon * X_pgd.grad.data.sign()
                elif i < change_point:
                    eta = first_step_size * X_pgd.grad.data.sign()
                else:
                    eta = step_size * X_pgd.grad.data.sign()
                X_pgd = X_pgd.detach() + eta.detach()
                X_pgd = torch.min(torch.max(X_pgd, X - epsilon), X + epsilon)
                X_pgd = Variable(torch.clamp(X_pgd, v_min, v_max), requires_grad=True)
            adv_check_and_update(X_pgd, model(X_pgd), y, not_correct, X_adv)

    adv_correct = (not_correct == 0).squeeze()
    return nat_correct, adv_correct, X_adv.detach().cpu().numpy()


def PGD_attack(model,
               X,
               y,
               epsilon=8. / 255.,
               num_steps=40,
               step_size=2. / 255.,
               num_random_starts=2,
               v_min=0.,
               v_max=1.):
    epsilon = epsilon * (v_max - v_min)
    step_size = step_size * (v_max - v_min)

    nat_logits = model(X)
    nat_pred = nat_logits.max(dim=1)[1]
    nat_correct = (nat_pred == y).squeeze()

    y_gt = one_hot_tensor(y, 10)

    not_correct = torch.zeros_like(y)
    X_adv = X.detach().clone()
    for _ in range(max(num_random_starts, 1)):
        X_pgd = Variable(X.data, requires_grad=True)
        if num_random_starts:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for i in range(num_steps):
            with torch.enable_grad():
                logits = model(X_pgd)
                loss = nn.CrossEntropyLoss(reduction='none')(logits, y)
                loss = torch.mean(loss)

                adv_check_and_update(X_pgd, logits, y, not_correct, X_adv)

            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = X_pgd.detach() + eta.detach()
            X_pgd = torch.min(torch.max(X_pgd, X - epsilon), X + epsilon)
            X_pgd = Variable(torch.clamp(X_pgd, v_min, v_max), requires_grad=True)
        adv_check_and_update(X_pgd, model(X_pgd), y, not_correct, X_adv)

    adv_correct = (not_correct == 0).squeeze()
    return nat_correct, adv_correct, X_adv.detach().cpu().numpy()
