# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils


def train(train_iter, dev_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    m_max = -99999
    whichmax = ''
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    output = open(os.path.join(args.save_dir, 'test.log'), "w+", encoding='utf-8')
    for attr, value in sorted(args.__dict__.items()):
        output.write("\t{}={} \n".format(attr.upper(), value))
        output.flush()
    output.write('----------------------------------------------------')
    output.flush()

    if args.lr_scheduler is not None:
        scheduler = None
        if args.lr_scheduler == 'lambda':
            lambda1 = lambda epoch: epoch // 30
            lambda2 = lambda epoch: 0.97 ** epoch
            scheduler = lr_scheduler.LambdaLR(optimizer, lambda2)
        elif args.lr_scheduler == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == '':
            pass

    steps = 0
    model.train()
    for epoch in range(1, args.epochs+1):

        if args.lr_scheduler is not None:
            scheduler.step()
            print(scheduler.get_lr())

        print("第", epoch, "次迭代")
        for batch in train_iter:
            feature, target = batch.text, batch.label
            # feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()

            if args.clip_norm is not None:
                utils.clip_grad_norm(model.parameters(), args.clip_norm)

            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                eval(dev_iter, model, args)

                m_str, acc = test(dev_iter, model, args)
                output.write(m_str + '-------' + str(steps))
                output.flush()

            if steps % args.save_interval == 0:
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

                m_str, acc = test(test_iter, model, args)
                output.write(m_str + '-------' + str(steps) + '\n')
                output.flush()
                if acc > m_max:
                    m_max = acc
                    whichmax = steps
    output.write('\nmax is {} using {}'.format(m_max, whichmax))
    output.flush()
    output.close()


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        size += batch.batch_size

    # size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))


def test(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

        size += batch.batch_size

    # size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    model.train()
    return '\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss, accuracy, corrects, size), accuracy


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]
