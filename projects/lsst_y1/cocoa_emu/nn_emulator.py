import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
import sys
from torchinfo import summary
from datetime import datetime
from numba import njit

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class Better_ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(Better_ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        #self.norm2 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm3 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#
        #self.act2 = #nn.Tanh()#nn.ReLU()#
        self.act3 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip             #(self.norm2(self.layer2(o1))) + xskip
        o3 = self.act3(self.norm3(o2))

        return o3

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions, dropout=False):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = self.drop(torch.reshape(prod,(batch_size,-1)))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions, dropout=False):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.in_size      = in_size
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = activation_fcn(in_size)  #nn.Tanh()   #nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)
        #self.act2         = nn.Tanh()#nn.ReLU()#
        #self.norm2        = torch.nn.BatchNorm1d(in_size)
        self.act3         = activation_fcn(in_size)  #nn.Tanh()
        self.norm3        = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights1 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        bias1 = torch.Tensor(in_size)
        self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        weights2 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        bias2 = torch.Tensor(in_size)
        self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        bound1 = 1 / np.sqrt(fan_in1) 
        nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        bound2 = 1 / np.sqrt(fan_in2) 
        nn.init.uniform_(self.bias2, -bound2, bound2)

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()

    def forward(self,x):
        mat1 = torch.block_diag(*self.weights1) # how can I do this on init rather than on each forward pass?
        mat2 = torch.block_diag(*self.weights2)

        o1 = self.act(self.norm(torch.matmul(x,mat1)+self.bias1))
        o2 = torch.matmul(o1,mat2)+self.bias2  #self.act2(self.norm2(torch.matmul(o1,mat2)+self.bias2))
        o3 = self.act3(self.norm3(o2+x))
        return o3

class activation_fcn(nn.Module):
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self,x):
        exp = torch.mul(self.beta,x)
        inv = torch.special.expit(exp)
        fac_2 = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out



class nn_pca_emulator:
    def __init__(self, 
                  model,
                  samples_mean, samples_std,
                  dv_fid, dv_std, cov_inv,
                  evecs,
                  device,
                  optim=None, lr=1e-3, reduce_lr=True, scheduler=None,
                  weight_decay=1e-3,
                  dtype='float'):
         
        self.optim        = optim
        self.device       = device 

        self.samples_mean = torch.Tensor(samples_mean)
        self.samples_std  = torch.Tensor(samples_std)
        self.dv_fid       = torch.Tensor(dv_fid)
        self.cov_inv      = torch.Tensor(cov_inv)
        self.dv_std       = torch.Tensor(dv_std)
        self.evecs        = evecs

        self.reduce_lr    = reduce_lr
        self.model        = model
        self.trained      = False
        self.weight_decay = weight_decay
        
        if self.optim is None:
            #print('Learning rate = {}'.format(lr))
            #print('Weight decay = {}'.format(weight_decay))
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr ,weight_decay=self.weight_decay)
        if self.reduce_lr == True:
            #print('Reduce LR on plateu: ',self.reduce_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',patience=15,factor=0.1)
        if dtype=='double':
            torch.set_default_dtype(torch.double)
            print('default data type = double')
        if device!=torch.device('cpu'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.generator = torch.Generator(device=self.device)

    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=1000, n_epochs=150):
        summary(self.model)
        #print('Batch size = ',batch_size)
        #print('N_epochs = ',n_epochs)

        # get normalization factors
        if not self.trained:
            self.X_mean = self.samples_mean
            self.X_std  = self.samples_std
            self.y_mean = self.dv_fid
            self.y_std  = self.dv_std

        # initialize arrays
        losses_train = []
        losses_vali = []
        loss = 100.

        # send everything to device
        self.model.to(self.device)
        tmp_y_std        = self.y_std.to(self.device)
        tmp_cov_inv      = self.cov_inv.to(self.device)
        tmp_X_mean       = self.X_mean.to(self.device)
        tmp_X_std        = self.X_std.to(self.device)
        tmp_X_validation = X_validation.to(self.device)#(X_validation.to(self.device) - tmp_X_mean)/tmp_X_std
        tmp_Y_validation = y_validation.to(self.device)#(y_validation.to(self.device) - tmp_X_mean)/tmp_X_std#y_validation.to(self.device)

        # Here is the input normalization
        X_train     = X #((X - self.X_mean)/self.X_std)
        y_train     = y #((y - self.X_mean)/self.X_std)#y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        validset    = torch.utils.data.TensorDataset(tmp_X_validation,tmp_Y_validation)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=self.generator)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=self.generator)
    
        #print('Datasets loaded!')
        print('Begin training...')
        train_start_time = datetime.now()
        for e in range(n_epochs):
            start_time = datetime.now()
            self.model.train()
            losses = []
            for i, data in enumerate(trainloader):    
                X       = data[0].to(self.device)
                Y_batch = data[1].to(self.device)
                Y_pred  = self.model(X) * tmp_y_std

                # PCA part
                diff = Y_batch - Y_pred
                loss1 = (diff \
                        @ tmp_cov_inv) \
                        @ torch.t(diff)

                ### remove the largest from each batch (ES TESTING!)
                # loss_arr = torch.diag(loss1)
                # sort_loss = torch.sort(loss_arr)
                # loss = torch.mean(sort_loss[0][:-5])
                ### END TESTING
                ### BEGIN TESTING
                #loss = torch.mean(torch.log(torch.diag(loss1))) # log chi^2
                #loss = torch.mean(torch.log(1+torch.diag(loss1))) # log hyperbola
                loss = torch.mean((1+2*torch.diag(loss1))**(1/2))-1 #hyperbola
                #loss = torch.mean((1+3*torch.diag(loss1))**(1/3))-1 #hyperbola^1/3
                ### END TESTING

                #loss = torch.mean(torch.diag(loss1)) # commented out for testing
                losses.append(loss.cpu().detach().numpy())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            losses_train.append(np.mean(losses))
            ###validation loss
            with torch.no_grad():
                self.model.eval()
                losses = []
                for i, data in enumerate(validloader):  
                    X_v       = data[0].to(self.device)
                    Y_v_batch = data[1].to(self.device)

                    Y_v_pred = self.model(X_v) * tmp_y_std

                    v_diff = Y_v_batch - Y_v_pred 
                    loss_vali1 = (v_diff \
                                    @ tmp_cov_inv) @ \
                                    torch.t(v_diff)

                    ### remove the largest 2 from each batch (ES TESTING!)
                    # loss_vali_arr = torch.diag(loss_vali1)
                    # sort_vali_loss = torch.sort(loss_vali_arr)
                    # loss_vali = torch.mean(sort_vali_loss[0][:-5])
                    ### END TESTING
                    ### BEGIN TESTING
                    #loss_vali = torch.mean(torch.log(torch.diag(loss_vali1))) # log chi^2
                    #loss_vali = torch.mean(torch.log(1+torch.diag(loss_vali1))) # log hyperbola
                    loss_vali = torch.mean((1+2*torch.diag(loss_vali1))**(1/2))-1 #hyperbola
                    #loss_vali = torch.mean((1+3*torch.diag(loss_vali1))**(1/3))-1 #hyperbola^1/3
                    ### END TESTING
                    
                    #loss_vali = torch.mean(torch.diag(loss_vali1)) # commented out and replaced with testing portion above
                    losses.append(np.float(loss_vali.cpu().detach().numpy()))

                losses_vali.append(np.mean(losses))
                if self.reduce_lr:
                    self.scheduler.step(losses_vali[e])

                self.optim.zero_grad()

            end_time = datetime.now()
            print('epoch {}, loss={:.5f}, validation loss={:.5f}, lr={:.2E} (epoch time: {:.1f})'.format(
                        e,
                        losses_train[-1],
                        losses_vali[-1],
                        self.optim.param_groups[0]['lr'],
                        (end_time-start_time).total_seconds()
                    ))#, total runtime: {} ({} average))
        
        np.savetxt("losses.txt", np.array([losses_train,losses_vali],dtype=np.float64))
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            y_pred = (self.model((X - self.X_mean) / self.X_std) * self.dv_std) #normalization

        y_pred = y_pred @ torch.linalg.inv(self.evecs)+ self.dv_fid # convert back to data basis
        return y_pred.cpu().detach().numpy()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
            f['evecs']  = self.evecs
        
    def load(self, filename, device=torch.device('cpu')):
        self.trained = True
        if device!=torch.device('cpu'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.model.load_state_dict(torch.load(filename,map_location=device))
        #summary(self.model)
        self.model.eval()
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:])
            self.X_std  = torch.Tensor(f['X_std'][:])
            self.dv_fid = torch.Tensor(f['dv_fid'][:])
            self.dv_std = torch.Tensor(f['dv_std'][:])
            self.evecs  = torch.Tensor(f['evecs'][:])

