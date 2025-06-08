import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import numpy as np
from torchvision.transforms.functional import rotate
import torchvision
import random
import datetime
import copy
import os
import gc
from torch.utils.tensorboard import SummaryWriter

from networks.generator import Generator
from networks.discriminator import Discriminator
from load_volume_data_RCA import Dataset
from samples_parameters import SAMPLES_PARA
import pickle
import time

ab_path = os.getcwd()
# ab_path = os.getcwd() + '/DeepCA/'
# ab_path_data = os.getcwd() + '/datasets/'
ab_path_data = ab_path + '/datasets/'

out_path=ab_path + '/outputs_results/checkpoints'
recons_out_path=ab_path+'/outputs_results/recons/'


if(not os.path.exists(out_path)):
    os.makedirs(out_path)

if(not os.path.exists(recons_out_path)):
    os.makedirs(recons_out_path)

do_eval=1
resume_model=1
LEARNING_RATE = 1e-4
MAX_EPOCHS = 200

BATCH_SIZE = 2

# Summary writer
run_folder = ab_path + '/runs/{date:%m_%d_%H:%M}'.format(date=datetime.datetime.now())
# writer = SummaryWriter(run_folder)

def set_torch():
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled= True
    torch.backends.cudnn.benmark= False
    torch.backends.cudnn.deterministic=True

def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_gradient_penalty(real_images, fake_images, discriminator, device, batch_size=BATCH_SIZE):
    eta = torch.FloatTensor(batch_size,2,1,1,1).uniform_(0,1).to(device)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))

    interpolated = eta * fake_images + ((1 - eta) * real_images)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    lambda_term = 10
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

def generation_eval(outputs,labels):
    l1_criterion = nn.L1Loss() #nn.MSELoss()

    l1_loss = l1_criterion(outputs, labels)

    return l1_loss

    
def do_evaluation_2(data_index, model, device, discriminator):
    model.eval()
    discriminator.eval()
    for id in data_index:
        data_file = ab_path_data + '/CCTA_BP/recon_' + str(id) + '.npy'
        input = np.transpose(np.load(data_file)[:,:,:,np.newaxis])
        label_file = ab_path_data + '/CCTA_GT/' + str(id) + '.npy'
        label = np.transpose(np.load(label_file)[:,:,:,np.newaxis])
        input = torch.from_numpy(input).float().to(device)
        label = torch.from_numpy(label).float().to(device)

        pre_time=time.time()
        output = model(input)
        rec=F.sigmoid(outputs)
        vessels=rec.cpu().numpy()
        next_time=time.time()
        run_time=next_time-pre_time
        print(f'run time of file index {id} is {run_time} seconds')
        outpath=ab_path + 'model_results/' 
        if(not os.path.exists(outpath)):
            os.makedirs(outpath)

        np.save(os.path.join(outpath, 'recon_' + str(id) + '.npy'), vessels)
        
        
def do_evaluation(dataloader, model, device, discriminator):
    model.eval()
    discriminator.eval()
    
    l1_losses = []
    G_losses = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    recons_data=[]
    projections=[]

    outpath = ab_path + '/model_results/' 

    if(not os.path.exists(outpath)):
        os.makedirs(outpath)
                
    with torch.no_grad():
        for data in dataloader:
            inputs, labels, ids = data[0].float().to(device), data[1].float().to(device), data[2]
            
            # calculate outputs by running images through the network
            pre_time=time.time()
            outputs=model(inputs)
            print('--------------------------------------')
            print(outputs.shape)
            rec=F.sigmoid(outputs)
            vessels=rec.cpu().numpy()
            next_time=time.time()
            run_time=next_time-pre_time

            for i in range(2):
                vessel=np.squeeze(vessels[i,0,:,:,:])
                lbl=labels[i,0,:,:,:].cpu().numpy()
                label=np.squeeze(lbl)
                # print(vessel.shape)
                # np.save(os.path.join(outpath, str(ids[i].cpu().numpy()) + '.npy'), vessel)
                np.savez_compressed(os.path.join(outpath, str(ids[i].cpu().numpy()) + '.npz'), vessel=vessel, label=label)

            print(f'run time is {run_time} ---------------------------------------')
            # recons_data.append(rec.cpu().numpy())
            # proj=inputs.cpu().numpy()
            # print(np.shape(proj))
            # projections.append(proj)
            
            DG_score = discriminator(torch.cat((inputs, F.sigmoid(outputs)), 1)).mean() # D(G(z))
            G_loss = -DG_score
            G_losses.append(G_loss.item())

            l1_loss = generation_eval(outputs,labels)
            l1_losses.append(l1_loss.item())
    
    # out_path=os.path.join(recons_out_path, 'result.pkl')
    # with open(out_path, 'wb') as f:
    #     res={'vessels': recons_data, 'projections': projections}
    #     pickle.dump(res, f)
        
    return np.mean(G_losses), np.mean(l1_losses)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Generator(in_channels=1, num_filters=SAMPLES_PARA['gen_filter_num'], class_num=1).to(device)
    discriminator = Discriminator(device, 2).to(device)
    #G and D optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(discriminator.parameters(), 
                                    lr=1e-4, betas=(0.5, 0.9))
    
    best_model_state = None
    best_D_model_state = None
    optimizer_state = None
    D_optimizer_state = None
    best_validation_loss = np.inf

    if(resume_model or do_eval==1):
        checkpoint = torch.load(ab_path + '/outputs_results/checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        start_epoch=checkpoint['epoch']+1
        best_model_state=checkpoint['best_model_state']
        best_D_model_state=checkpoint['best_D_model_state']
        early_stop_count_val=checkpoint['early_stop_count_val']
        train_index=checkpoint['train_index']
        validation_index=checkpoint['validation_index']
        test_index=checkpoint['test_index']

    else:
        start_epoch=0
        early_stop_count_val = 0
        train_index=SAMPLES_PARA['train_index']
        validation_index=SAMPLES_PARA['validation_index']
        test_index=SAMPLES_PARA['test_index']

    batch_size = BATCH_SIZE

    #Dataset setup
    training_set = Dataset(train_index)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    val_set = Dataset(validation_index)
    validationloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    test_set = Dataset(test_index)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * (-1)
    num_train_divide = np.floor(SAMPLES_PARA["num_train_data"]/batch_size)
    num_critics = 2
    
    if(do_eval==0):
        writer = SummaryWriter(run_folder)
        for epoch in range(start_epoch, MAX_EPOCHS):
            gc.collect()
            torch.cuda.empty_cache()
    
            model.train()
            discriminator.train()
            l1_losses = []
            D_losses = []
            D_losses_cur = []
            G_losses = []
            combined_losses = []
            Wasserstein_Ds = []
            Wasserstein_Ds_cur= []
    
            for i, data in enumerate(trainloader, 0):
                torch.cuda.empty_cache()
    
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].float().to(device), data[1].float().to(device)
    
                ######################## CCTA/VG training
                ####### adversarial loss
                # Requires grad, Generator requires_grad = False
                for p in discriminator.parameters():
                    p.requires_grad = True
                for p in model.parameters():
                    p.requires_grad = False
                gc.collect()
                torch.cuda.empty_cache()
    
                D_optimizer.zero_grad()
                outputs = model(inputs)
    
                # Classify the generated and real batch images
                DX_score = discriminator(torch.cat((inputs, labels), 1)).mean() # D(x)
    
                DG_score = discriminator(torch.cat((inputs, outputs), 1).detach()).mean() # D(G(z))
    
                # Train with gradient penalty
                gradient_penalty = calculate_gradient_penalty(torch.cat((inputs, labels), 1), torch.cat((inputs, outputs), 1).detach(), discriminator, device)
    
                D_loss = (DG_score - DX_score + gradient_penalty)
                Wasserstein_D = DX_score - DG_score
    
                # Update parameters
                D_loss.backward()
                D_optimizer.step()
                D_losses.append(D_loss.detach().item())
                D_losses_cur.append(D_loss.detach().item())
                Wasserstein_Ds.append(Wasserstein_D.detach().item())
                Wasserstein_Ds_cur.append(Wasserstein_D.detach().item())
                ####################            
    
                ###### generator loss
                # Generator update
                if (i+1) % num_critics == 0:
                    for p in discriminator.parameters():
                        p.requires_grad = False  # to avoid computation
                    for p in model.parameters():
                        p.requires_grad = True
                    gc.collect()
                    torch.cuda.empty_cache()
    
                    optimizer.zero_grad()
                    outputs = model(inputs)
    
                    DG_score = discriminator(torch.cat((inputs, outputs), 1)).mean() # D(G(z))
                    G_loss = -DG_score
                    G_losses.append(G_loss.detach().item())
    
                    l1_loss = generation_eval(outputs,labels)
                    l1_losses.append(l1_loss.detach().item())
    
                    ###################
                    combined_loss = G_loss + l1_loss*100
                    combined_losses.append(combined_loss.detach().item())
    
                    # update parameters
                    combined_loss.backward()
                    optimizer.step()
    
                    writer.add_scalar('Loss_iter/l1_3d', l1_loss.detach(), epoch*num_train_divide+i+1)
                    writer.add_scalar('Loss_iter/G_loss', G_loss.detach(), epoch*num_train_divide+i+1)
                    writer.add_scalar('Loss_iter/D_loss', np.mean(D_losses_cur), epoch*num_train_divide+i+1)
                    writer.add_scalar('Loss_iter/Wasserstein_D', np.mean(Wasserstein_Ds_cur), epoch*num_train_divide+i+1)
                    writer.add_scalar('Loss_iter/combined_loss', combined_loss.detach(), epoch*num_train_divide+i+1)
                    writer.add_scalars('Loss_iter/G_D_loss', {'G_loss': G_loss.detach(), 'D_loss': np.mean(D_losses_cur)}, epoch*num_train_divide+i+1)
    
                    D_losses_cur = []
                    Wasserstein_Ds_cur = []
    
            #do validation
            G_loss_val, l1_loss_val = do_evaluation(validationloader, model, device, discriminator)
            combined_loss_val = G_loss_val + l1_loss_val*100
            validation_loss = l1_loss_val
    
            writer.add_scalar('Loss/train', np.mean(Wasserstein_Ds), epoch+1)
            writer.add_scalar('Loss/l1_3d', np.mean(l1_losses), epoch+1)
            writer.add_scalar('Loss/D_loss', np.mean(D_losses), epoch+1)
            writer.add_scalar('Loss/G_loss', np.mean(G_losses), epoch+1)
            writer.add_scalar('Loss/combined_losses', np.mean(combined_losses), epoch+1)
    
            writer.add_scalar('Loss/l1_3d_val', l1_loss_val, epoch+1)
            writer.add_scalar('Loss/G_loss_val', G_loss_val, epoch+1)
            writer.add_scalar('Loss/combined_losses_val', combined_loss_val, epoch+1)
    
            writer.add_scalars('Loss/l1_3d_tv', {'train': np.mean(l1_losses), 'validation': l1_loss_val}, epoch+1)
            writer.add_scalars('Loss/G_loss_tv', {'train': np.mean(G_losses), 'validation': G_loss_val}, epoch+1)
            writer.add_scalars('Loss/combined_losses_tv', {'train': np.mean(combined_losses), 'validation': combined_loss_val}, epoch+1)
    
            writer.add_scalars('Loss/G_D_loss', {'G_loss': np.mean(G_losses), 'D_loss': np.mean(D_losses)}, epoch+1)
    
            print(f"training epoch {epoch}")
            if (epoch + 1) % 1 == 0:
                model.eval()
                torch.save(
                        {
                            "network": model.state_dict(),
                            "discriminator": discriminator.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "D_optimizer": D_optimizer.state_dict(),
                            "epoch": epoch,
                            "early_stop_count_val":early_stop_count_val,
                            "train_index":SAMPLES_PARA['train_index'],
                            "validation_index":SAMPLES_PARA['validation_index'],
                            "test_index":SAMPLES_PARA['test_index'],
                            "best_model_state":best_model_state,
                            "best_D_model_state":best_D_model_state
                        },
                        # ab_path + 'outputs_results/checkpoints/Epoch_' + str(epoch+1) + '.tar',
                        ab_path + '/outputs_results/checkpoint.pth',
    
                    )
    
    
            # early stopping if validation loss is increasing or staying the same after five epoches
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                early_stop_count_val = 0
    
                # # Save a checkpoint of the best validation loss model so far
                # # print("saving this best validation loss model so far!")
                best_model_state = copy.deepcopy(model.state_dict())
                best_D_model_state = copy.deepcopy(discriminator.state_dict())
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                D_optimizer_state = copy.deepcopy(D_optimizer.state_dict()) 
            else:
                early_stop_count_val += 1
                # print('no improvement on validation at this epoch, continue training...')
    
            if early_stop_count_val >= 20:
                print('early stopping validation!!!')
                break
            
    # evaluate on test set
    print('\n############################### testing evaluation on best trained model so far')
    model.load_state_dict(best_model_state)
    discriminator.load_state_dict(best_D_model_state)
    G_loss_test, l1_loss_test = do_evaluation(testloader, model, device, discriminator)
    test_loss = G_loss_test + l1_loss_test*100

    # print('\n############################### generating 3d vessels for test data ')
    # do_evaluation_2(test_index, model, device, discriminator)
    
    print('Testdataset Evaluation - test loss: {0:3.8f}, G loss: {1:3.8f}, l1 loss: {2:3.8f}'
                .format(test_loss, G_loss_test.item(), l1_loss_test.item()))

if __name__ == '__main__':
    # set_torch()
    set_random_seed(1, False)

    main()
