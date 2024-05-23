'''
Code for calculating curvature as per HIKER 
Author: Isha Garg, 2023.
Modified by Deepak Ravikumar Tatachar, 2023
'''

import torch
import torch.nn as nn
import torch.nn.parallel

class CurveScore():
    def __init__(self, dataset, model, device, temp):
        self.model = model
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss() 
        self.num_classes= dataset.num_classes
        self.last_logits = None 
        self.train_loader = dataset.train_loader
        self.test_loader = dataset.test_loader
        self.device = device
        self.temp = temp

    def find_z(self, net, inputs, targets, h) :
        '''
        Finding the direction in the regularizer
        '''
        net = net.eval()
        inputs.requires_grad_()
        _ = net(inputs)
        loss_z = self.criterion(net(inputs), targets)                
        
        #loss_z.backward(torch.ones(targets.size()).to(self.device))         
        loss_z.backward()
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)  
        if inputs.grad is not None:
          inputs.grad.zero_() 
        net.zero_grad()
        return z, norm_grad

    def regularizer(self, net, inputs, targets, pred_labels, h=1e-3, niter=10):
        '''
        Regularizer term in CURE
        '''
        net = net.eval()
        # z, norm_grad =self.find_z(net, inputs, targets, h )
        regs = torch.zeros(inputs.shape[0])
        eigs = torch.zeros(inputs.shape[0])
        for _ in range(niter):
            v = torch.randint_like(inputs, high=2).to(self.device)
            
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = 1.*(h) * (v+1e-7) #/ (v.reshape(v.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)  
            inputs.requires_grad_()
            outputs_pos = net(inputs + v)
            outputs_orig = net(inputs)
            if pred_labels:
                targets = torch.argmax(outputs_orig, 1).to(torch.long)
            loss_pos = self.criterion(outputs_pos/self.temp, targets)
            loss_orig = self.criterion(outputs_orig/self.temp, targets)
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs)[0]
            regs += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
            eigs += torch.diag(torch.matmul(v.reshape(inputs.shape[0],-1), grad_diff.reshape(inputs.shape[0],-1).T)).cpu().detach()   #vTHv = trace of H = sum of eigenvals
            net.zero_grad()
            if inputs.grad is not None:
              inputs.grad.zero_()
        return regs/niter, eigs/niter

    def score_true_labels_with_comparison(self, pred_labels=False):
        scores = torch.zeros(self.dataset.train_length)
        self.model.eval()
        scores, labels = self.rank_samples_true_labels_with_comparison(pred_labels=pred_labels)
        self.last_logits = labels
        return scores, labels

    def rank_samples_true_labels_with_comparison(self, pred_labels):
        score_curvature = torch.zeros(self.dataset.train_length)
        labels = torch.zeros(self.dataset.train_length)
        idx = 0
        for _, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True
            self.model.zero_grad()
            stop_idx = inputs.shape[0] + idx
            score_ours_sq, _ = self.regularizer(self.model, inputs, targets, pred_labels)
            score_curvature[idx:stop_idx] = score_ours_sq.cpu().detach()
            labels[idx:stop_idx] = targets.cpu().detach()
            idx = stop_idx
        return score_curvature, labels
