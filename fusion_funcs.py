import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import logging
from torch.nn import functional as F
from torch.utils import tensorboard
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import cv2
from PIL import Image
import gc
import pandas as pd
from utils import load_model, metric, adjust_learning_rate, save_trainable_checkpoint, load_trainable_checkpoint, metric


class TrainingModel:

    def __init__(self, fold_id, net, img_cont_loss, global_clip_loss, local_clip_loss, soft_clip_loss, clas_loss, loader_train, loader_val, loader_test, config, optimizer, scheduler=None):

        self.logger = logging.getLogger("FusionModel")
        self.fold_id = fold_id
        self.model = net
        self.img_cont_loss = img_cont_loss
        self.global_clip_loss = global_clip_loss
        self.local_clip_loss = local_clip_loss
        self.soft_clip_loss = soft_clip_loss
        self.clas_loss = clas_loss
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler(device="cuda" if config.cuda else "cpu")
        self.device = torch.device("cuda" if config.cuda else "cpu")
        self.lam_global = config.lam_global
        self.lam_local = config.lam_local
        self.lam_soft = config.lam_soft
        if hasattr(config, 'lam_cls'):
            self.lam_cls = config.lam_cls
        
        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            load_trainable_checkpoint(self.model, config.pretrained_path + '_fold_' + str(fold_id) + '.pth')

        self.model.to(self.device)
        self.img_cont_loss.to(self.device)
        self.global_clip_loss.to(self.device)
        self.local_clip_loss.to(self.device)
        self.clas_loss.to(self.device)

    def pretraining(self):
        print("Pretraining started")
        print(self.img_cont_loss)
        print(self.global_clip_loss)
        print(self.local_clip_loss)
        print(self.clas_loss)
        print(self.optimizer)
        print(self.scheduler)
        best_val_loss = 1000
        sum_writer = tensorboard.SummaryWriter(os.path.join(self.config.output_dir, "fold_" + str(self.fold_id))) # type: ignore

        for epoch in range(self.config.nb_epochs):

            train_img_cont_loss, train_global_loss, train_local_loss, train_soft_loss = 0, 0, 0, 0
            val_img_cont_loss, val_global_loss, val_local_loss, val_soft_loss = 0, 0, 0, 0

            if self.scheduler is None:
                adjust_learning_rate(self.optimizer, epoch, self.config.lr, self.config.nb_epochs, self.config.warm_epochs, self.config.power_lr)
            current_lr = self.optimizer.param_groups[0]['lr']
            print("Epoch: {}, Learning Rate: {}".format(epoch, current_lr))

            self.model.train()
            nb_batches = len(self.loader_train)
            train_loss = 0
            pbar = tqdm(desc="Training", total=nb_batches, mininterval=20)
            
            for batch in self.loader_train:
                pbar.update()
                input_1 = batch['image_1'].to(self.device, non_blocking=True)
                input_2 = batch['image_2'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                local_text = batch['local_text']
                attr_label = batch['attr_label'].to(self.device, non_blocking=True)
                # global_text = batch['global_text']
                self.optimizer.zero_grad()

                with autocast(device_type='cuda' if self.config.cuda else 'cpu'):
                    return_dict = self.model(input_1, input_2, local_text)
                    if self.config.cont_loss == "AdaSP":
                        features = torch.cat([return_dict['img_1_cls'], return_dict['img_2_cls']], dim=1)
                        img_cont_loss = self.img_cont_loss(features)
                    else:
                        loss_1, loss_2, logits, target = self.img_cont_loss(return_dict['img_1_cls'], return_dict['img_2_cls'], labels)
                        img_cont_loss = (loss_1 + loss_2) / 2
                    if self.config.global_clip_loss == "Unsupervised":
                        global_loss_1, global_loss_2 = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        global_loss = (global_loss_1 + global_loss_2) / 2
                    else:
                        if self.config.enable_fusion:
                            global_loss = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        else:
                            global_loss_1 = self.global_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'])
                            global_loss_2 = self.global_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'])
                            global_loss = (global_loss_1 + global_loss_2) / 2
                    
                    if self.config.enable_fusion:
                        soft_loss = self.soft_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'], attr_label, attr_label)
                    else:
                        soft_loss_1 = self.soft_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss_2 = self.soft_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss = (soft_loss_1 + soft_loss_2) / 2

                    if self.config.enable_fusion:
                        local_loss = self.local_clip_loss(return_dict['fusion_patches'], return_dict['text_patches'])
                    else:
                        local_loss_1 = self.local_clip_loss(return_dict['img_1_patches'], return_dict['text_patches'])
                        local_loss_2 = self.local_clip_loss(return_dict['img_2_patches'], return_dict['text_patches'])
                        local_loss = (local_loss_1 + local_loss_2) / 2
                    batch_loss = (1 - self.lam_global - self.lam_local - self.lam_soft) * img_cont_loss + self.lam_global * global_loss + self.lam_local * local_loss + self.lam_soft * soft_loss
                    
                self.scaler.scale(batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                train_img_cont_loss += float(img_cont_loss.item()) / nb_batches
                train_global_loss += float(global_loss.item()) / nb_batches
                train_local_loss += float(local_loss.item()) / nb_batches
                train_soft_loss += float(soft_loss.item()) / nb_batches
                train_loss += float(batch_loss.item()) / nb_batches
            pbar.close()


            nb_batch = len(self.loader_val)
            pbar = tqdm(desc="Validation", total=nb_batch, mininterval=20)
            val_loss = 0

            with torch.no_grad():
                self.model.eval()
                for batch in self.loader_val:
                    pbar.update()
                    input_1 = batch['image_1'].to(self.device, non_blocking=True)
                    input_2 = batch['image_2'].to(self.device, non_blocking=True)
                    labels = batch['label'].to(self.device, non_blocking=True)
                    local_text = batch['local_text']
                    attr_label = batch['attr_label'].to(self.device, non_blocking=True)
                    # global_text = batch['global_text']

                    return_dict = self.model(input_1, input_2, local_text)
                    if self.config.img_cont_loss == "AdaSP":
                        features = torch.cat([return_dict['img_1_cls'], return_dict['img_2_cls']], dim=1)
                        img_cont_loss = self.img_cont_loss(features)
                    else:
                        loss_1, loss_2, logits, target = self.img_cont_loss(return_dict['img_1_cls'], return_dict['img_2_cls'], labels)
                        img_cont_loss = (loss_1 + loss_2) / 2
                    if self.config.global_clip_loss == "Unsupervised":
                        global_loss_1, global_loss_2 = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        global_loss = (global_loss_1 + global_loss_2) / 2
                    else:
                        if self.config.enable_fusion:
                            global_loss = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        else:
                            global_loss_1 = self.global_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'])
                            global_loss_2 = self.global_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'])
                            global_loss = (global_loss_1 + global_loss_2) / 2
                    if self.config.enable_fusion:
                        soft_loss = self.soft_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'], attr_label, attr_label)
                    else:
                        soft_loss_1 = self.soft_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss_2 = self.soft_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss = (soft_loss_1 + soft_loss_2) / 2
                    if self.config.enable_fusion:
                        local_loss = self.local_clip_loss(return_dict['fusion_patches'], return_dict['text_patches'])
                    else:
                        local_loss_1 = self.local_clip_loss(return_dict['img_1_patches'], return_dict['text_patches'])
                        local_loss_2 = self.local_clip_loss(return_dict['img_2_patches'], return_dict['text_patches'])
                        local_loss = (local_loss_1 + local_loss_2) / 2
                    batch_loss = (1 - self.lam_global - self.lam_local - self.lam_soft) * img_cont_loss + self.lam_global * global_loss + self.lam_local * local_loss + self.lam_soft * soft_loss
                    val_img_cont_loss += float(img_cont_loss.item()) / nb_batch
                    val_global_loss += float(global_loss.item()) / nb_batch
                    val_local_loss += float(local_loss.item()) / nb_batch
                    val_soft_loss += float(soft_loss.item()) / nb_batch
                    val_loss += float(batch_loss.item()) / nb_batch
            pbar.close()

            print("\nEpoch [{}/{}] Train: img cont loss = {:.4f}, global loss = {:.4f}, local loss = {:.4f}, soft loss = {:.4f}, loss = {:.4f}, Val: img cont loss = {:.4f}, global loss = {:.4f}, local loss = {:.4f}, soft loss = {:.4f}, loss = {:.4f}".format(
                epoch, self.config.nb_epochs, train_img_cont_loss, train_global_loss, train_local_loss, train_soft_loss, train_loss, val_img_cont_loss, val_global_loss, val_local_loss, val_soft_loss, val_loss), flush=True)
            sum_writer.add_scalar('train/img_cont_loss', train_img_cont_loss, epoch)
            sum_writer.add_scalar('train/global_loss', train_global_loss, epoch)
            sum_writer.add_scalar('train/local_loss', train_local_loss, epoch)
            sum_writer.add_scalar('train/soft_loss', train_soft_loss, epoch)
            sum_writer.add_scalar('train/loss', train_loss, epoch)
            sum_writer.add_scalar('valid/img_cont_loss', val_img_cont_loss, epoch)
            sum_writer.add_scalar('valid/global_loss', val_global_loss, epoch)
            sum_writer.add_scalar('valid/local_loss', val_local_loss, epoch)
            sum_writer.add_scalar('valid/soft_loss', val_soft_loss, epoch)
            sum_writer.add_scalar('valid/loss', val_loss, epoch)
            
            # if (epoch % self.config.nb_epochs_per_saving == 0 or epoch == self.config.nb_epochs - 1) and epoch > 0:
            #     # trainable_state_dict = {
            #     #         k: v.cpu() for k, v in self.model.state_dict().items()
            #     #         if not k.startswith("text_model.")
            #     #     }
            #     torch.save({
            #         "epoch": epoch,
            #         "model": self.model.state_dict(),
            #         # "optimizer": self.optimizer.state_dict()
            #         },
            #         os.path.join(self.config.output_dir, "{name}_epoch_{epoch}_fold_{fold}.pth".
            #                      format(name="FusionModel_Pretrained", fold=self.fold_id, epoch=epoch)))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Best model!")
                # trainable_state_dict = {
                #     k: v.cpu() for k, v in self.model.state_dict().items()
                #     if not k.startswith("text_model.")
                # }
                save_path = os.path.join(self.config.output_dir, "FusionModel_Pretrained_Best_fold_{fold}.pth".format(fold=self.fold_id))
                save_trainable_checkpoint(self.model, save_path, epoch)


    def finetuning(self):
        print("Finetuning started")
        print(self.img_cont_loss)
        print(self.global_clip_loss)
        print(self.local_clip_loss)
        print(self.clas_loss)
        print("Lam_cls: {}".format(self.lam_cls))
        print(self.optimizer)
        print(self.scheduler)
        best_val_loss, best_bacc, best_f1 = 1000, 0, 0
        sum_writer = tensorboard.SummaryWriter(os.path.join(self.config.output_dir, "fold_" + str(self.fold_id))) # type: ignore
        best_acc, best_es_acc, best_es_auc, best_es_class_attr_scaled_f1 = 0, 0, 0, 0

        for epoch in range(self.config.nb_epochs):

            if self.scheduler is None:
                adjust_learning_rate(self.optimizer, epoch, self.config.lr, self.config.nb_epochs, self.config.warm_epochs, self.config.power_lr)
            current_lr = self.optimizer.param_groups[0]['lr']
            print("Epoch: {}, Learning Rate: {}".format(epoch, current_lr))

            self.model.train()
            nb_batches = len(self.loader_train)
            train_img_cont_loss, train_global_loss, train_local_loss, train_soft_loss, train_clas_loss, train_loss = 0, 0, 0, 0, 0, 0
            pbar = tqdm(desc="Training", total=nb_batches, mininterval=20)
            
            for batch in self.loader_train:
                pbar.update()
                input_1 = batch['image_1'].to(self.device, non_blocking=True)
                input_2 = batch['image_2'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                local_text = batch['local_text']
                attr_label = batch['attr_label'].to(self.device, non_blocking=True)
                sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None
                # global_text = batch['global_text']
                self.optimizer.zero_grad()

                with autocast(device_type='cuda' if self.config.cuda else 'cpu'):
                    return_dict = self.model(input_1, input_2, local_text)
                    if self.config.cont_loss == "AdaSP":
                        features = torch.cat([return_dict['img_1_cls'], return_dict['img_2_cls']], dim=1)
                        img_cont_loss = self.img_cont_loss(features)
                    else:
                        loss_1, loss_2, logits, target = self.img_cont_loss(return_dict['img_1_cls'], return_dict['img_2_cls'], labels)
                        img_cont_loss = (loss_1 + loss_2) / 2
                    if self.config.global_clip_loss == "Unsupervised":
                        global_loss_1, global_loss_2 = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        global_loss = (global_loss_1 + global_loss_2) / 2
                    else:
                        if self.config.enable_fusion:
                            global_loss = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        else:
                            global_loss_1 = self.global_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'])
                            global_loss_2 = self.global_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'])
                            global_loss = (global_loss_1 + global_loss_2) / 2
                    if self.config.enable_fusion:
                        soft_loss = self.soft_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'], attr_label, attr_label)
                    else:
                        soft_loss_1 = self.soft_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss_2 = self.soft_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss = (soft_loss_1 + soft_loss_2) / 2
                    if self.config.enable_fusion:
                        local_loss = self.local_clip_loss(return_dict['fusion_patches'], return_dict['text_patches'])
                    else:
                        local_loss_1 = self.local_clip_loss(return_dict['img_1_patches'], return_dict['text_patches'])
                        local_loss_2 = self.local_clip_loss(return_dict['img_2_patches'], return_dict['text_patches'])
                        local_loss = (local_loss_1 + local_loss_2) / 2
                    batch_loss = (1 - self.lam_global - self.lam_local - self.lam_soft) * img_cont_loss + self.lam_global * global_loss + self.lam_local * local_loss + self.lam_soft * soft_loss
                    clas_loss_1 = self.clas_loss(return_dict['cls_logits_1'], labels)
                    clas_loss_2 = self.clas_loss(return_dict['cls_logits_2'], labels)
                    clas_loss = (clas_loss_1 + clas_loss_2) / 2
                    batch_loss = (1 - self.lam_cls) * batch_loss + self.lam_cls * (clas_loss_1 + clas_loss_2) / 2

                self.scaler.scale(batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                train_img_cont_loss += float(img_cont_loss.item()) / nb_batches
                train_global_loss += float(global_loss.item()) / nb_batches
                train_local_loss += float(local_loss.item()) / nb_batches
                train_soft_loss += float(soft_loss.item()) / nb_batches
                train_clas_loss += float(clas_loss.item()) / nb_batches
                train_loss += float(batch_loss.item()) / nb_batches
            pbar.close()


            nb_batch = len(self.loader_val)
            pbar = tqdm(desc="Validation", total=nb_batch, mininterval=20)
            val_img_cont_loss, val_global_loss, val_local_loss, val_soft_loss, val_clas_loss, val_loss, correct = 0, 0, 0, 0, 0, 0, 0
            prob_list, label_list, sens_attr_list = [], [], []

            with torch.no_grad():
                self.model.eval()
                for batch in self.loader_val:
                    pbar.update()
                    input_1 = batch['image_1'].to(self.device, non_blocking=True)
                    input_2 = batch['image_2'].to(self.device, non_blocking=True)
                    labels = batch['label'].to(self.device, non_blocking=True)
                    local_text = batch['local_text']
                    sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None
                    attr_label = batch['attr_label'].to(self.device, non_blocking=True)
                    # global_text = batch['global_text']
                    
                    return_dict = self.model(input_1, input_2, local_text)
                    if self.config.img_cont_loss == "AdaSP":
                        features = torch.cat([return_dict['img_1_cls'], return_dict['img_2_cls']], dim=1)
                        img_cont_loss = self.img_cont_loss(features)
                    else:
                        loss_1, loss_2, logits, target = self.img_cont_loss(return_dict['img_1_cls'], return_dict['img_2_cls'], labels)
                        img_cont_loss = (loss_1 + loss_2) / 2
                    if self.config.global_clip_loss == "Unsupervised":
                        global_loss_1, global_loss_2 = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        global_loss = (global_loss_1 + global_loss_2) / 2
                    else:
                        if self.config.enable_fusion:
                            global_loss = self.global_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'])
                        else:
                            global_loss_1 = self.global_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'])
                            global_loss_2 = self.global_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'])
                            global_loss = (global_loss_1 + global_loss_2) / 2
                    if self.config.enable_fusion:
                        soft_loss = self.soft_clip_loss(return_dict['fusion_cls'], return_dict['text_cls'], attr_label, attr_label)
                    else:
                        soft_loss_1 = self.soft_clip_loss(return_dict['img_1_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss_2 = self.soft_clip_loss(return_dict['img_2_cls'], return_dict['text_cls'], attr_label, attr_label)
                        soft_loss = (soft_loss_1 + soft_loss_2) / 2
                    if self.config.enable_fusion:
                        local_loss = self.local_clip_loss(return_dict['fusion_patches'], return_dict['text_patches'])
                    else:
                        local_loss_1 = self.local_clip_loss(return_dict['img_1_patches'], return_dict['text_patches'])
                        local_loss_2 = self.local_clip_loss(return_dict['img_2_patches'], return_dict['text_patches'])
                        local_loss = (local_loss_1 + local_loss_2) / 2
                    batch_loss = (1 - self.lam_global - self.lam_local - self.lam_soft) * img_cont_loss + self.lam_global * global_loss + self.lam_local * local_loss + self.lam_soft * soft_loss
                    clas_loss_1 = self.clas_loss(return_dict['cls_logits_1'], labels)
                    clas_loss_2 = self.clas_loss(return_dict['cls_logits_2'], labels)
                    clas_loss = (clas_loss_1 + clas_loss_2) / 2
                    batch_loss = (1 - self.lam_cls) * batch_loss + self.lam_cls * clas_loss
                    val_img_cont_loss += float(img_cont_loss.item()) / nb_batch
                    val_global_loss += float(global_loss.item()) / nb_batch
                    val_local_loss += float(local_loss.item()) / nb_batch
                    val_soft_loss += float(soft_loss.item()) / nb_batch
                    val_clas_loss += float(clas_loss.item()) / nb_batch
                    val_loss += float(batch_loss.item()) / nb_batch

                    prob_1 = F.softmax(return_dict['cls_logits_1'], dim=1)
                    prob_2 = F.softmax(return_dict['cls_logits_2'], dim=1)
                    prob_list.append(((prob_1 + prob_2) / 2).cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if sens_attr is not None:
                        sens_attr_list.append(sens_attr.cpu().numpy())
                    pred_1 = return_dict['cls_logits_1'].argmax(dim=1, keepdim=True)
                    pred_2 = return_dict['cls_logits_2'].argmax(dim=1, keepdim=True)
                    correct += (pred_1.eq(labels.view_as(pred_1)).sum().item() + pred_2.eq(labels.view_as(pred_2)).sum().item()) / 2
                    
            
            accuracy = correct / len(self.loader_val.dataset)
            return_dict = metric(np.concatenate(prob_list, axis=0), np.concatenate(label_list, axis=0), np.concatenate(sens_attr_list, axis=0) if len(sens_attr_list) > 0 else None, equity_alpha=self.config.equity_alpha)
            bacc = return_dict['overall_bacc']
            f1 = return_dict['overall_classification_report']['macro avg']['f1-score']

            if len(sens_attr_list) > 0:
                ES_acc = return_dict['overall_ES_acc']
                ES_auc = return_dict['overall_ES_AUC']
                ES_class_attr_scaled_f1 = return_dict['overall_ES_Class_Attr_Scaled_F1']
            else:
                ES_acc = None
                ES_auc = None
                ES_class_attr_scaled_f1 = None
            pbar.close()

            sum_writer.add_scalar('train/img_cont_loss', train_img_cont_loss, epoch)
            sum_writer.add_scalar('train/global_loss', train_global_loss, epoch)
            sum_writer.add_scalar('train/local_loss', train_local_loss, epoch)
            sum_writer.add_scalar('train/soft_loss', train_soft_loss, epoch)
            sum_writer.add_scalar('train/clas_loss', train_clas_loss, epoch)
            sum_writer.add_scalar('train/loss', train_loss, epoch)
            sum_writer.add_scalar('valid/img_cont_loss', val_img_cont_loss, epoch)
            sum_writer.add_scalar('valid/global_loss', val_global_loss, epoch)
            sum_writer.add_scalar('valid/local_loss', val_local_loss, epoch)
            sum_writer.add_scalar('valid/soft_loss', val_soft_loss, epoch)
            sum_writer.add_scalar('valid/clas_loss', val_clas_loss, epoch)
            sum_writer.add_scalar('valid/loss', val_loss, epoch)
            sum_writer.add_scalar('valid/accuracy', accuracy, epoch)
            sum_writer.add_scalar('valid/bacc', bacc, epoch)
            sum_writer.add_scalar('valid/f1', f1, epoch)
            if len(sens_attr_list) > 0:
                sum_writer.add_scalar('valid/ES_acc', ES_acc, epoch)
                sum_writer.add_scalar('valid/ES_auc', ES_auc, epoch)
                sum_writer.add_scalar('valid/ES_class_attr_scaled_f1', ES_class_attr_scaled_f1, epoch)

            print("\nEpoch [{}/{}] Train: img cont loss = {:.4f}, global loss = {:.4f}, local loss = {:.4f}, soft loss = {:.4f}, clas loss = {:.4f}, loss = {:.4f}, Val: img cont loss = {:.4f}, global loss = {:.4f}, local loss = {:.4f}, soft loss = {:.4f}, clas loss = {:.4f}, loss = {:.4f}, accuracy = {:.4f}, bacc = {:.4f}, f1 = {:.4f}".format(
                epoch, self.config.nb_epochs, train_img_cont_loss, train_global_loss, train_local_loss, train_soft_loss, train_clas_loss,
                train_loss, val_img_cont_loss, val_global_loss, val_local_loss, val_soft_loss, val_clas_loss, val_loss, accuracy, bacc, f1), flush=True)
            if len(sens_attr_list) > 0:
                print("Val: ES_acc = {:.4f}, ES_auc = {:.4f}, ES_class_attr_scaled_f1 = {:.4f}".format(ES_acc, ES_auc, ES_class_attr_scaled_f1), flush=True)
            
            # if (epoch % self.config.nb_epochs_per_saving == 0 or epoch == self.config.nb_epochs - 1) and epoch > 0:
            #     # trainable_state_dict = {
            #     #         k: v.cpu() for k, v in self.model.state_dict().items()
            #     #         if not k.startswith("text_model.")
            #     #     }
            #     torch.save({
            #         "epoch": epoch,
            #         "model": self.model.state_dict(),
            #         # "optimizer": self.optimizer.state_dict()
            #         },
            #         os.path.join(self.config.output_dir, "{name}_epoch_{epoch}_fold_{fold}.pth".
            #                      format(name="FusionModel_Finetuned", fold=self.fold_id, epoch=epoch)))
            
            if ES_class_attr_scaled_f1 is not None and ES_class_attr_scaled_f1 > best_es_class_attr_scaled_f1:
                best_es_class_attr_scaled_f1 = ES_class_attr_scaled_f1
                print("Best model!")
                save_path = os.path.join(self.config.output_dir, "FusionModel_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id))
                save_trainable_checkpoint(self.model, save_path, epoch)
            elif ES_class_attr_scaled_f1 is None:
                if self.config.metric == "f1" and f1 > best_f1:
                    best_f1 = f1
                    print("Best model!")
                    save_path = os.path.join(self.config.output_dir, "FusionModel_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id))
                    save_trainable_checkpoint(self.model, save_path, epoch)    
                elif self.config.metric == "bacc" and bacc > best_bacc:
                    best_bacc = bacc
                    print("Best model!")
                    save_path = os.path.join(self.config.output_dir, "FusionModel_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id))
                    save_trainable_checkpoint(self.model, save_path, epoch)    
                elif self.config.metric == "acc" and accuracy > best_acc:
                    best_acc = accuracy
                    print("Best model!")
                    save_path = os.path.join(self.config.output_dir, "FusionModel_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id))
                    save_trainable_checkpoint(self.model, save_path, epoch)


    def test(self):
        load_trainable_checkpoint(self.model, os.path.join(self.config.output_dir, "FusionModel_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id)))
        nb_batch = len(self.loader_test)
        pbar = tqdm(desc="Testing", total=nb_batch, mininterval=20)
        self.model.eval()
        predict_dict, truth_dict, count_dict, ensemble_dict, ensemble_prob_dict, sens_attr_dict = {}, {}, {}, {}, {}, {}
        case_correct, img_correct = 0, 0
        name_list, predict_list, label_list, prob_list, sens_attr_list = [], [], [], [], []

        with torch.no_grad():
            for batch in self.loader_test:
                pbar.update()
                inputs = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                local_text = batch['local_text']
                attr_label = batch['attr_label'].to(self.device, non_blocking=True)
                # global_text = batch['global_text']
                name = batch['name'][0]
                sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None

                patient = name.split('/')[0]
                if patient not in predict_dict:
                    predict_dict[patient] = []
                if patient not in truth_dict:
                    truth_dict[patient] = labels.cpu().numpy()[0]
                if patient not in count_dict:
                    count_dict[patient] = 0
                if patient not in sens_attr_dict and sens_attr is not None and sens_attr.cpu().numpy()[0] != -1:
                    sens_attr_dict[patient] = sens_attr.cpu().numpy()[0]
                count_dict[patient] += 1

                return_dict = self.model(inputs, text=local_text)
                pred = return_dict['logits_1'].argmax(dim=1, keepdim=True)
                prob = F.softmax(return_dict['logits_1'], dim=1)
                img_correct += pred.eq(labels.view_as(pred)).sum().item()
                name_list.append(name)
                if sens_attr is not None and sens_attr.cpu().numpy()[0] != -1:
                    sens_attr_list.append(sens_attr.cpu().numpy()[0])
                predict_list.append(torch.flatten(pred).cpu().numpy()[0])
                label_list.append(torch.flatten(labels).cpu().numpy()[0])
                prob_list.append(torch.flatten(prob).cpu().numpy())\
                
                predict_dict[patient].append(torch.flatten(prob).cpu().numpy())

        for patient in predict_dict.keys():
            ensemble_prob = np.mean(np.stack(predict_dict[patient]), axis=0)
            ensemble = np.argmax(ensemble_prob, axis=0)
            ensemble_dict[patient] = int(ensemble)
            ensemble_prob_dict[patient] = ensemble_prob
            if ensemble == truth_dict[patient]:
                case_correct += 1
        case_acc = case_correct / len(predict_dict)
        img_acc = img_correct / len(self.loader_test.dataset)
        case_result_dict = metric(list(ensemble_prob_dict.values()), list(truth_dict.values()), list(sens_attr_dict.values()) if len(sens_attr_dict) > 0 else None, group_wise=True, equity_alpha=self.config.equity_alpha)
        case_f1 = case_result_dict['overall_classification_report']['macro avg']['f1-score']
        case_auc = case_result_dict['overall_auc']
        case_bacc = case_result_dict['overall_bacc']
        case_con_matrix = case_result_dict['overall_confusion_matrix']
        case_class_report = case_result_dict['overall_classification_report']
        img_result_dict = metric(prob_list, label_list, sens_attr_list if len(sens_attr_list) > 0 else None, group_wise=True, equity_alpha=self.config.equity_alpha) 
        img_f1 = img_result_dict['overall_classification_report']['macro avg']['f1-score']
        img_auc = img_result_dict['overall_auc']
        img_bacc = img_result_dict['overall_bacc']
        img_con_matrix = img_result_dict['overall_confusion_matrix']
        img_class_report = img_result_dict['overall_classification_report']
        pbar.close()

        csv_dict = {
            'case_0_f1': case_class_report['0']['f1-score'], # type: ignore
            'case_1_f1': case_class_report['1']['f1-score'], # type: ignore
            'case_2_f1': case_class_report['2']['f1-score'] if '2' in case_class_report else None,
            'case_3_f1': case_class_report['3']['f1-score'] if '3' in case_class_report else None,
            'case_4_f1': case_class_report['4']['f1-score'] if '4' in case_class_report else None,
            'case_5_f1': case_class_report['5']['f1-score'] if '5' in case_class_report else None,
            'case_f1': case_f1,
            'case_acc': case_acc,
            'case_bacc': case_bacc,
            'case_auc': case_auc,
            'img_0_f1': img_class_report['0']['f1-score'], # type: ignore
            'img_1_f1': img_class_report['1']['f1-score'], # type: ignore
            'img_2_f1': img_class_report['2']['f1-score'] if '2' in img_class_report else None,
            'img_3_f1': img_class_report['3']['f1-score'] if '3' in img_class_report else None,
            'img_4_f1': img_class_report['4']['f1-score'] if '4' in img_class_report else None,
            'img_5_f1': img_class_report['5']['f1-score'] if '5' in img_class_report else None,
            'img_f1': img_f1,
            'img_acc': img_acc,
            'img_bacc': img_bacc,
            'img_auc': img_auc,
        }
        
        if len(sens_attr_list) > 0:
            csv_dict['case_ES_acc'] = case_result_dict['overall_ES_acc']
            csv_dict['case_ES_AUC'] = case_result_dict['overall_ES_AUC']
            csv_dict['case_ES_Class_Attr_Scaled_F1'] = case_result_dict['overall_ES_Class_Attr_Scaled_F1']
            csv_dict['case_EO'] = case_result_dict['overall_EO']
            csv_dict['case_DP'] = case_result_dict['overall_DP']
            csv_dict['case_attr_0_fungal_f1'] = case_result_dict['attr_0_classification_report']['0']['f1-score']
            csv_dict['case_attr_0_bacterial_f1'] = case_result_dict['attr_0_classification_report']['1']['f1-score']
            csv_dict['case_attr_0_f1'] = case_result_dict['attr_0_classification_report']['macro avg']['f1-score']
            csv_dict['case_attr_0_AUC'] = case_result_dict['attr_0_AUC']
            csv_dict['case_attr_0_acc'] = case_result_dict['attr_0_classification_report']['accuracy']
            csv_dict['case_attr_1_fungal_f1'] = case_result_dict['attr_1_classification_report']['0']['f1-score']
            csv_dict['case_attr_1_bacterial_f1'] = case_result_dict['attr_1_classification_report']['1']['f1-score']
            csv_dict['case_attr_1_f1'] = case_result_dict['attr_1_classification_report']['macro avg']['f1-score']
            csv_dict['case_attr_1_AUC'] = case_result_dict['attr_1_AUC']
            csv_dict['case_attr_1_acc'] = case_result_dict['attr_1_classification_report']['accuracy']
            csv_dict['img_ES_acc'] = img_result_dict['overall_ES_acc']
            csv_dict['img_ES_AUC'] = img_result_dict['overall_ES_AUC']
            csv_dict['img_ES_Class_Attr_Scaled_F1'] = img_result_dict['overall_ES_Class_Attr_Scaled_F1']
            csv_dict['img_EO'] = img_result_dict['overall_EO']
            csv_dict['img_DP'] = img_result_dict['overall_DP']
            csv_dict['img_attr_0_fungal_f1'] = img_result_dict['attr_0_classification_report']['0']['f1-score']
            csv_dict['img_attr_0_bacterial_f1'] = img_result_dict['attr_0_classification_report']['1']['f1-score']
            csv_dict['img_attr_0_f1'] = img_result_dict['attr_0_classification_report']['macro avg']['f1-score']
            csv_dict['img_attr_0_AUC'] = img_result_dict['attr_0_AUC']
            csv_dict['img_attr_0_acc'] = img_result_dict['attr_0_classification_report']['accuracy']
            csv_dict['img_attr_1_fungal_f1'] = img_result_dict['attr_1_classification_report']['0']['f1-score']
            csv_dict['img_attr_1_bacterial_f1'] = img_result_dict['attr_1_classification_report']['1']['f1-score']
            csv_dict['img_attr_1_f1'] = img_result_dict['attr_1_classification_report']['macro avg']['f1-score']
            csv_dict['img_attr_1_AUC'] = img_result_dict['attr_1_AUC']
            csv_dict['img_attr_1_acc'] = img_result_dict['attr_1_classification_report']['accuracy']
            
        fold_col = f'fold_{self.fold_id}'
        csv_save_path = os.path.join(self.config.output_dir, 'test_results.csv')
        df_to_save = pd.DataFrame(csv_dict, index=[0]).T # type: ignore
        df_to_save.columns = [fold_col]
        if not os.path.exists(csv_save_path):
            df_to_save.to_csv(csv_save_path, mode='w', header=True)
        else:
            existing_df = pd.read_csv(csv_save_path, index_col=0)
            existing_df[fold_col] = df_to_save[fold_col]
            existing_df.to_csv(csv_save_path, mode='w', header=True)
        print("Test results saved to {}".format(csv_save_path))

        out_file = open(os.path.join(self.config.out_file_path + '_fold_{}'.format(self.fold_id) + '.txt'), 'w', encoding='utf-8')
        result = "Case Result: Accuracy = {:.4f}, F1-score = {:.4f}, BACC = {:.4f}, AUC = {:.4f}\n".format(case_acc, case_f1, case_bacc, case_auc)
        print('\n' + result)
        out_file.write(result)
        case_con_matrix = case_con_matrix.tolist()
        print('Confusion matrix: ')
        out_file.write('Confusion matrix: ' + '\n')
        for l in case_con_matrix:
            print(l)
            out_file.write(str(l) + '\n')
        print('Classification Report: ')
        out_file.write('Classification Report: ' + '\n')
        print(case_class_report)
        out_file.write(str(case_class_report)) # type: ignore
        if len(sens_attr_list) > 0:
            es_result = "Equalized Odds (EO) = {:.4f}, Demographic Parity (DP) = {:.4f}\n".format(
                case_result_dict['overall_EO'], case_result_dict['overall_DP'])
            print(es_result)
            out_file.write(es_result)
            eq_acc_auc = "Equalized Accuracy = {:.4f}, Equalized AUC = {:.4f}, Equalized Class-Attr Scaled F1 = {:.4f}\n".format(
                case_result_dict['overall_ES_acc'], case_result_dict['overall_ES_AUC'], case_result_dict['overall_ES_Class_Attr_Scaled_F1'])
            print(eq_acc_auc)
            out_file.write(eq_acc_auc)
            
        title = '\nName\tpredict\ttruth\tattr\n'
        print(title)
        out_file.write(title)
        for patient in predict_dict.keys():
            content = str(patient) + '\t' + str(ensemble_dict[patient]) + '\t' + str(truth_dict[patient])
            if len(sens_attr_list) > 0:
                content += '\t' + str(sens_attr_dict[patient])
            else:
                content += '\t-1'
            print(content)
            out_file.write(content + '\n')

        result = "Image Result: Accuracy = {:.4f}, F1-score = {:.4f}\n".format(img_acc, img_f1)
        print('\n' + result)
        out_file.write(result)
        img_con_matrix = img_con_matrix.tolist()
        print('Confusion matrix: ')
        out_file.write('Confusion matrix: ' + '\n')
        for l in img_con_matrix:
            print(l)
            out_file.write(str(l) + '\n')
        print('Classification Report: ')
        out_file.write('Classification Report: ' + '\n')
        print(img_class_report)
        out_file.write(str(img_class_report)) # type: ignore
        
        if len(sens_attr_list) > 0:
            es_result = "Equalized Odds (EO) = {:.4f}, Demographic Parity (DP) = {:.4f}\n".format(
                img_result_dict['overall_EO'], img_result_dict['overall_DP'])
            print(es_result)
            out_file.write(es_result)
            eq_acc_auc = "Equalized Accuracy = {:.4f}, Equalized AUC = {:.4f}, Equalized Class-Attr Scaled F1 = {:.4f}\n".format(
                img_result_dict['overall_ES_acc'], img_result_dict['overall_ES_AUC'], img_result_dict['overall_ES_Class_Attr_Scaled_F1'])
            print(eq_acc_auc)
            out_file.write(eq_acc_auc)
        
        title = '\nName\tprobability\tpredict\ttruth\tsens_attr\n'
        print(title)
        out_file.write(title)
        for i in range(len(name_list)):
            content = str(name_list[i]) + '\t' + str(prob_list[i]) + '\t' + str(predict_list[i]) + '\t' + str(label_list[i])
            if len(sens_attr_list) > 0:
                content += '\t' + str(sens_attr_list[i])
            else:
                content += '\t-1'
            print(content)
            out_file.write(content + '\n')        
        out_file.close()
        
