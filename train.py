import torch
import time
from tensorboardX import SummaryWriter
from utils import *
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.autograd import Variable
import time
import gc

from objective import *


def train(model, dataloaders_dict, criterion, optimizer, val_interval, num_epoch, load_model_bool, model_dir, run_name,
          unsupervised, args, beta, device):
    print('Unsupervised learning: ', unsupervised)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    exp_dir = 'runs_{}'.format(args.exp)
    if args.demo:
        run_name += '_demo'
    if args.unsup_num is not None:
        run_name += '_unsup_{}'.format(args.unsup_num)
    run_name += '_' + current_time
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(exp_dir + '/' + run_name):
        os.mkdir(exp_dir + '/' + run_name)
        with open(exp_dir + '/' + run_name + '/config.txt', 'w') as f:
            f.write(str(args))
        print(f'New directory {run_name} created')
    save_dir = exp_dir + '/' + run_name
    # Parameters used between epochs
    starting_epoch = 0
    epoch_CE_loss = {'train': None, 'val': None}
    epoch_KLD_loss = {'train': None, 'val': None}
    epoch_loss = {'train': None, 'val': None}
    epoch_acc = {'train': None, 'val': None}
    epoch_avg_acc = {'train': None, 'val': None}
    epoch_kappa = {'train': None, 'val': None}
    epoch_mse = {'train': None, 'val': None}
    epoch_cm = {'train': None, 'val': None}
    min_loss_checkpoint = {'val_loss': None, 'best_dir': None,
                           'lastest_dir': None, 'val_overall_acc': None,
                           'kappa_score': None, 'val_avg_acc': None,
                           'val_kappa': None, 'val_mse': None}
    # Load (saved) model to device
    if load_model_bool:
        model, starting_epoch = load_model(model_dir, min_loss_checkpoint, device)
        progress_info_writer(save_dir, "Load model from {}".format(model_dir))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Tensorboard logging
    writer_CE_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/CE/train')
    writer_KLD_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/KLD/train')
    writer_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/train')
    writer_CE_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/CE/val')
    writer_KLD_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/KLD/val')
    writer_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/val')

    debugger = 0

    criterion_unsup = EntLoss(args, args.lam1, args.lam2, pqueue=None)
    criterion_unsup.to(device)

    for epoch in range(starting_epoch, starting_epoch + num_epoch):
        for phase in ['train', 'val']:  # Either train or val
            if phase == 'val' and epoch % val_interval != 0:  # Not the epoch for val
                epoch_loss['val'] = None
                epoch_acc['val'] = None
                continue

            num_batch = len(dataloaders_dict[phase])
            print('Number of Batch:', num_batch)
            print('Length Of Dataset', len(dataloaders_dict[phase].dataset))
            running_loss = 0.0
            running_CE_loss = 0.0
            running_KLD_loss = 0.0
            running_corrects = 0
            num_sample = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to eval

            # Iterate batches
            bar = tqdm(total=len(dataloaders_dict[phase]), desc='Processing', ncols=90)
            truths = []
            pred_labels = []
            if unsupervised:
                #kl_loss = torch.nn.KLDivLoss()
                # CMD 20211118 --> matching the UDA implementation in github
                kl_loss = torch.nn.KLDivLoss(reduction='none')
                log_prob = torch.nn.LogSoftmax(dim=1)
                prob = torch.nn.Softmax(dim=1)
            for batch, ((inputs, labels, _, _), (image_1, image_2)) in enumerate(
                    zip(dataloaders_dict[phase], dataloaders_dict['unsup'])):

                if debugger == 0:
                    print('DEBUGGER: input min, max, mean, shape', inputs.min(), inputs.max(), inputs.mean(), inputs.size())
                    debugger = 1

                inputs = inputs.to(device)
                labels = labels.to(device)
                num_sample += labels.shape[0]
                image_1 = image_1.to(device)
                image_2 = image_2.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    sup_loss = criterion(outputs, labels)
                    #print('@@@@', sup_loss, outputs, inputs.mean(), inputs.std())
                    if unsupervised:
                        
                        if 0:
                            # CMD 20211118 --> matching the UDA implementation in github
                            # "Sharpening Predictions of the original image"
                            ori_logits = model(image_1)
                            if args.uda_softmax_temp != -1:
                                ori_logits_tgt = ori_logits / args.uda_softmax_temp
                            else:
                                ori_logits_tgt = ori_logits

                            #logit_1 = model(image_1)
                            logit_2 = model(image_2)
                            unsup_loss_persample = kl_loss(log_prob(ori_logits_tgt), prob(logit_2))
                            
                            # "Confidence based masking of the original image"
                            if args.uda_confidence_thresh != -1:
                                ori_prob = prob(ori_logits)
                                #print('$$$'*10, ori_prob)
                                largest_prob, inds = torch.max(ori_prob, dim=1)#.values
                                #print ('largest_prob',largest_prob)
                                mask = largest_prob > args.uda_confidence_thresh
                                #print('mask',mask)
                                #print('unsup_loss_persample',unsup_loss_persample)
                                #print('unsup_loss_persample',unsup_loss_persample[mask])
                                if torch.any(mask):
                                    unsup_loss = torch.mean(unsup_loss_persample[mask])
                                else:
                                    unsup_loss = torch.tensor(0)
                        else:
                            ### add loss from TWIST paper
                            logit_1 = model(image_1)
                            logit_2 = model(image_2)
                            tmp_loss = criterion_unsup(logit_1,logit_2)
                            unsup_loss = tmp_loss['final']
                            #print('Unsup Loss:', tmp_loss['final'], ' KL: ', tmp_loss['kl'], ' eh: ',tmp_loss['eh'], ' he: ',tmp_loss['he'] )

                    else:
                        unsup_loss = torch.tensor(0)
                    
                    loss = sup_loss + beta * unsup_loss

                    preds = torch.argmax(outputs, 1)
                    if phase == 'train':  # Update in train phase
                        loss.backward()
                        optimizer.step()
                truths.append(labels.reshape(-1, 1).data.cpu().numpy())
                pred_labels.append(preds.reshape(-1, 1).data.cpu().numpy())

                running_CE_loss += sup_loss.item() * inputs.shape[0]
                running_KLD_loss += unsup_loss.item() * beta * inputs.shape[0]
                running_loss += loss.item() * inputs.shape[0]  # Batch loss
                running_corrects += torch.sum(preds == labels).item()
                #bar.update(1)
                if phase == 'train':
                    # progress_info(epoch, starting_epoch + num_epoch, batch, num_batch, batch_speed, epoch_loss, epoch_acc)
                    if batch % 200 == 0 or batch == num_batch - 1:
                        avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
                        output = 'Epoch {}: {:} CE Loss:{:.4g}; KLD Loss:{:.4g}; Accuracy:{}/{}={:.4g}; Average Acc:{:4g}; Kappa:{:4g}; MSE:{:4g}; lr:{:4g}' \
                            .format(epoch, phase, running_CE_loss / num_sample,
                                    running_KLD_loss / num_sample,
                                    running_corrects,
                                    num_sample,
                                    running_corrects / num_sample,
                                    avg_acc,
                                    kappa,
                                    mse,
                                    args.learning_rate)
                        progress_info_writer(save_dir, output)
                if phase == 'val' and batch == num_batch - 1:
                    avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
                    output1 = '*' * 20
                    output = 'Epoch {}: {:} CE Loss:{:.4g}; KLD Loss:{:.4g}; Accuracy:{}/{}={:.4g}; Average Acc:{:4g}; Kappa:{:4g}; MSE:{:4g}' \
                        .format(epoch, phase, running_CE_loss / num_sample,
                                running_KLD_loss / num_sample,
                                running_corrects,
                                num_sample,
                                running_corrects / num_sample,
                                avg_acc,
                                kappa,
                                mse)
                    progress_info_writer(save_dir, output1, output, str(cm), output1)
            avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
            epoch_CE_loss[phase] = running_CE_loss / len(dataloaders_dict[phase].dataset)
            epoch_KLD_loss[phase] = running_KLD_loss / len(dataloaders_dict[phase].dataset)
            epoch_loss[phase] = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc[phase] = running_corrects / len(dataloaders_dict[phase].dataset)
            epoch_avg_acc[phase] = avg_acc
            print('###################')
            print(running_corrects / len(dataloaders_dict[phase].dataset), running_corrects / num_sample)
            print('###################')
            if phase == 'train':
                if unsupervised:
                    writer_CE_train.add_scalar('loss', epoch_CE_loss['train'], epoch)
                    writer_KLD_train.add_scalar('loss', epoch_KLD_loss['train'], epoch)
                writer_train.add_scalar('loss', epoch_loss['train'], epoch)
                writer_train.add_scalar('accuracy', epoch_acc['train'], epoch)
            else:
                save_model(min_loss_checkpoint, epoch_loss, epoch_acc, epoch_avg_acc, model.cpu(), epoch, save_dir)
                model.to(device)
                if unsupervised:
                    writer_CE_val.add_scalar('loss', epoch_CE_loss['val'], epoch)
                    writer_KLD_val.add_scalar('loss', epoch_KLD_loss['val'], epoch)
                writer_val.add_scalar('loss', epoch_loss['val'], epoch)
                writer_val.add_scalar('accuracy', epoch_acc['val'], epoch)

    writer_train.close()
    writer_val.close()
    writer_CE_train.close()
    writer_KLD_train.close()
    writer_CE_val.close()
    writer_KLD_val.close()
    print('Training Complete!' + ' ' * 10)


def train_mt(model, dataloaders_dict, criterion, optimizer, val_interval, num_epoch, load_model_bool, model_dir, run_name,
          unsupervised, args, beta, device):
    print('Unsupervised learning: ', unsupervised)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    exp_dir = 'runs_{}'.format(args.exp)
    if args.demo:
        run_name += '_demo'
    run_name += '_' + current_time
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(exp_dir + '/' + run_name):
        os.mkdir(exp_dir + '/' + run_name)
        with open(exp_dir + '/' + run_name + '/config.txt', 'w') as f:
            f.write(str(args))
        print(f'New directory {run_name} created')
    save_dir = exp_dir + '/' + run_name
    # Parameters used between epochs
    starting_epoch = 0
    epoch_CE_loss = {'train': None, 'val': None}
    epoch_KLD_loss = {'train': None, 'val': None}
    epoch_loss = {'train': None, 'val': None}
    epoch_acc = {'train': None, 'val': None}
    epoch_avg_acc = {'train': None, 'val': None}
    epoch_kappa = {'train': None, 'val': None}
    epoch_mse = {'train': None, 'val': None}
    epoch_cm = {'train': None, 'val': None}
    min_loss_checkpoint = {'val_loss': None, 'best_dir': None,
                           'lastest_dir': None, 'val_overall_acc': None,
                           'kappa_score': None, 'val_avg_acc': None,
                           'val_kappa': None, 'val_mse': None}
    # Load (saved) model to device
    if load_model_bool:
        model, starting_epoch = load_model(model_dir, min_loss_checkpoint, device)
        progress_info_writer(save_dir, "Load model from {}".format(model_dir))

    # Tensorboard logging
    writer_CE_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/CE/train')
    writer_KLD_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/KLD/train')
    writer_train = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/train')
    writer_CE_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/CE/val')
    writer_KLD_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/KLD/val')
    writer_val = SummaryWriter(log_dir=exp_dir + "/" + run_name + '/log/val')

    for epoch in range(starting_epoch, starting_epoch + num_epoch):
        for phase in ['train', 'val']:  # Either train or val
            if phase == 'val' and epoch % val_interval != 0:  # Not the epoch for val
                epoch_loss['val'] = None
                epoch_acc['val'] = None
                continue

            num_batch = len(dataloaders_dict[phase])
            print('Number of Batch:', num_batch)
            print('Length Of Dataset', len(dataloaders_dict[phase].dataset))
            running_loss = 0.0
            running_CE_loss = 0.0
            running_KLD_loss = 0.0
            running_corrects = 0
            num_sample = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to eval

            # Iterate batches
            bar = tqdm(total=len(dataloaders_dict[phase]), desc='Processing', ncols=90)
            truths = []
            pred_labels = []
            if unsupervised:
                kl_loss = torch.nn.KLDivLoss()
                log_prob = torch.nn.LogSoftmax(dim=1)
                prob = torch.nn.Softmax(dim=1)
            for batch, (inputs, labels, _, inputs_aug) in enumerate(
                    dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                num_sample += labels.shape[0]
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print('@@@@', num_sample, 'XXX', outputs)
                    sup_loss = criterion(outputs, labels)
                    if unsupervised:
                        inputs_aug = inputs_aug.to(device)
                        logit_2 = model(inputs_aug)
                        unsup_loss = kl_loss(log_prob(outputs), prob(logit_2))
                    else:
                        unsup_loss = torch.tensor(0, device=device)
                    loss = sup_loss + beta * unsup_loss
                    preds = torch.argmax(outputs, 1)
                    if phase == 'train':  # Update in train phase
                        loss.backward()
                        optimizer.step()
                truths.append(labels.reshape(-1, 1).data.cpu().numpy())
                pred_labels.append(preds.reshape(-1, 1).data.cpu().numpy())

                running_CE_loss += sup_loss.item() * inputs.shape[0]
                running_KLD_loss += unsup_loss.item() * beta * inputs.shape[0]
                running_loss += loss.item() * inputs.shape[0]  # Batch loss
                running_corrects += torch.sum(preds == labels).item()
                bar.update(1)
                if phase == 'train':
                    # progress_info(epoch, starting_epoch + num_epoch, batch, num_batch, batch_speed, epoch_loss, epoch_acc)
                    if batch % 5 == 0 or batch == num_batch - 1:
                        avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
                        output = 'Epoch {}: {:} CE Loss:{:.4g}; KLD Loss:{:.4g}; Accuracy:{}/{}={:.4g}; Average Acc:{:4g}; Kappa:{:4g}; MSE:{:4g}' \
                            .format(epoch, phase, running_CE_loss / num_sample,
                                    running_KLD_loss / num_sample,
                                    running_corrects,
                                    num_sample,
                                    running_corrects / num_sample,
                                    avg_acc,
                                    kappa,
                                    mse)
                        progress_info_writer(save_dir, output)
                if phase == 'val' and batch == num_batch - 1:
                    avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
                    output1 = '*' * 20
                    output = 'Epoch {}: {:} CE Loss:{:.4g}; KLD Loss:{:.4g}; Accuracy:{}/{}={:.4g}; Average Acc:{:4g}; Kappa:{:4g}; MSE:{:4g}' \
                        .format(epoch, phase, running_CE_loss / num_sample,
                                running_KLD_loss / num_sample,
                                running_corrects,
                                num_sample,
                                running_corrects / num_sample,
                                avg_acc,
                                kappa,
                                mse)
                    progress_info_writer(save_dir, output1, output, str(cm), output1)
            avg_acc, mse, kappa, cm = compute_metrics(truths, pred_labels)
            epoch_CE_loss[phase] = running_CE_loss / len(dataloaders_dict[phase].dataset)
            epoch_KLD_loss[phase] = running_KLD_loss / len(dataloaders_dict[phase].dataset)
            epoch_loss[phase] = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc[phase] = running_corrects / len(dataloaders_dict[phase].dataset)
            epoch_avg_acc[phase] = avg_acc
            print('###################')
            print(running_corrects / len(dataloaders_dict[phase].dataset), running_corrects / num_sample)
            print('###################')
            if phase == 'train':
                if unsupervised:
                    writer_CE_train.add_scalar('loss', epoch_CE_loss['train'], epoch)
                    writer_KLD_train.add_scalar('loss', epoch_KLD_loss['train'], epoch)
                writer_train.add_scalar('loss', epoch_loss['train'], epoch)
                writer_train.add_scalar('accuracy', epoch_acc['train'], epoch)
            else:
                save_model(min_loss_checkpoint, epoch_loss, epoch_acc, epoch_avg_acc, model.cpu(), epoch, save_dir)
                model.to(device)
                if unsupervised:
                    writer_CE_val.add_scalar('loss', epoch_CE_loss['val'], epoch)
                    writer_KLD_val.add_scalar('loss', epoch_KLD_loss['val'], epoch)
                writer_val.add_scalar('loss', epoch_loss['val'], epoch)
                writer_val.add_scalar('accuracy', epoch_acc['val'], epoch)

    writer_train.close()
    writer_val.close()
    writer_CE_train.close()
    writer_KLD_train.close()
    writer_CE_val.close()
    writer_KLD_val.close()
    print('Training Complete!' + ' ' * 10)


def validate_epoch(net, val_loader, criterion, use_cuda = True,loss_type='CE'):
    net.train(False)
    running_loss = 0.0
    sm = nn.Softmax(dim=1)

    truth = []
    preds = []
    bar = tqdm(total=len(val_loader), desc='Processing', ncols=90)
    names_all = []
    n_batches = len(val_loader)
    for i, (batch, targets, names, _) in enumerate(val_loader):
        #print('XXXXXX',batch.min(), batch.max(),names)
        if use_cuda:
            if loss_type == 'CE':
                labels = Variable(targets.long().cuda())
                inputs = Variable(batch.cuda())
            elif loss_type == 'MSE':
                labels = Variable(targets.float().cuda())
                inputs = Variable(batch.cuda())
        else:
            if loss_type == 'CE':
                labels = Variable(targets.float())
                inputs = Variable(batch)
            elif loss_type == 'MSE':
                labels = Variable(targets.float())
                inputs = Variable(batch)

        outputs = net(inputs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        if loss_type =='CE':
            probs = sm(outputs).data.cpu().numpy()
        elif loss_type =='MSE':
            probs = outputs
            probs[probs < 0] = 0
            probs[probs > 4] = 4
            probs = probs.view(1,-1).squeeze(0).round().data.cpu().numpy()
        preds.append(probs)
        truth.append(targets.cpu().numpy())
        names_all.extend(names)
        running_loss += loss.item()
        bar.update(1)
        gc.collect()
    gc.collect()
    bar.close()
    if loss_type =='CE':
        preds = np.vstack(preds)
    else:
        preds = np.hstack(preds)
    truth = np.hstack(truth)
    return running_loss / n_batches, preds, truth, names_all