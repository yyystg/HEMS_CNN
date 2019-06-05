import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

def train(args,train_loader, model, optimizer,criterion,test_loader,train_loss_loader,writer,epoch):
    steps = (epoch-1)*train_loader.__len__()

    for idx, batch in enumerate(train_loader):
        model.train()
        feature, target = batch[0], batch[1]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align????
        feature = preprocessing(feature,args.t)

        if not args.no_cuda:
            feature, target = feature.float().cuda(), target.float().cuda()

        optimizer.zero_grad()
        logit = model(feature)

        #print('logit vector', logit.size())
        #print('target vector', target.size())
        loss = criterion(logit, target)
        loss.backward()
        optimizer.step()

        steps += 1
        if steps % args.log_interval == 0:
            test_loss, test_mae_loss, prediction_mae, prediction_rmse = eval(args, test_loader, model, criterion)
            train_loss, train_mae_loss, _, _ = eval(args, train_loss_loader, model, criterion)
            write_energy2tensorboard(train_loss, test_loss, prediction_rmse, steps, writer, name='mse')
            write_energy2tensorboard(train_mae_loss, test_mae_loss, prediction_mae, steps, writer, name='_mae')
            sys.stdout.write(
                '\rBatch[{}] - loss: {:.6f}'.format(steps,loss.item()))

def eval(args,data_loader,model,criterion):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            feature, target = batch[0], batch[1]
            feature = preprocessing(feature,args.t)
            if not args.no_cuda:
                feature, target = feature.float().cuda(), target.float().cuda()

            logit = model(feature)
            loss = criterion(logit,target)
            mae_loss = F.l1_loss(logit,target).cuda()
            if args.output_normalization:
                logit = denormalize(logit,data_loader)
                target = denormalize(target,data_loader)
            mae_loss = F.l1_loss(logit, target).cuda()
            mae = MAE(logit,target)
            rmse = RMSE(logit,target)

    print('\nEvaluation - loss: {:.6f}'.format(loss))
    return loss,mae_loss,mae,rmse

def denormalize(out,data_loader):
    mins = data_loader.dataset.out_min
    maxs = data_loader.dataset.out_max
    out = np.asarray(out.cpu())

    out_de = out*(maxs-mins)+mins
    out_de = torch.from_numpy(out_de).cuda()
    return out_de

def MAE(output,target):
    err = abs(output-target)
    err = torch.mean(err,dim=0)
    return err

def RMSE(output,target):
    err = output-target
    err = err*err
    err = torch.mean(err,dim=0)
    err = err.sqrt()
    return err

def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epochs_{}.pth.tar'.format(save_prefix, epochs)
    torch.save({
                'epoch': epochs + 1,
                'state_dict': model.state_dict(),
            }, save_path)

def preprocessing(input,t):
    input_edit = input[:,:t,:].numpy()#0혹은 -1으로 채우기
    input_edit2 = input[:,t:,:-5].numpy()
    input_edit2 = np.concatenate((input_edit2,np.zeros((len(input),31-t,5))),axis=2)
    input_edit = np.concatenate((input_edit,input_edit2),axis=1)
    input_edit = torch.tensor(input_edit)
    return input_edit

def write_energy2tensorboard(train_loss,test_loss,err,epoch,writer,name=""):
    writer.add_scalars('Loss'+name,{'train loss': train_loss,'test loss': test_loss},epoch)
    writer.add_scalars('energy_prediction_mae'+name,{'elec':err[0],'water':err[1],'gas':err[2],'hot_water':err[3],'heating':err[4]},epoch)