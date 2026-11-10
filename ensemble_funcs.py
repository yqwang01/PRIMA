import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import logging
from torch.utils import tensorboard
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from losses import selfpaced_weighted
from utils import adjust_learning_rate, metric, save_trainable_checkpoint, load_trainable_checkpoint, metric
import pandas as pd


class TrainingModel:

    def __init__(self, fold_id, net, ense_loss, loader_train, loader_val, loader_test, config, optimizer, scheduler=None):
        """

        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("MultiModaCL")
        self.ense_loss = ense_loss
        self.model = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.scaler = GradScaler('cuda' if config.cuda else 'cpu')
        self.loader = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.fold_id = fold_id
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.lam_mag = config.lam_mag
        self.metrics = {}

        self.model.to(self.device)
        self.ense_loss.to(self.device)

    def _get_logits(self, images, texts=None):
        outputs = self.model(images, texts)
        if isinstance(outputs, (tuple, list)):
            if len(outputs) >= 2:
                return outputs[1] if self.config.loss_mode == "target" else outputs[0]
            return outputs[0]  # pyright: ignore[reportGeneralTypeIssues]
        return outputs


    def train(self):
        print(self.ense_loss)
        print(self.optimizer)
        print(self.scheduler)
        best_acc, best_bacc, best_f1, best_es_acc, best_es_auc, best_es_class_attr_scaled_f1 = 0, 0, 0, 0, 0, 0
        sum_writer = tensorboard.SummaryWriter(os.path.join(self.config.output_dir, "fold_" + str(self.fold_id))) # type: ignore
        
        for epoch in range(self.config.nb_epochs):
            
            if self.scheduler is None:
                adjust_learning_rate(self.optimizer, epoch, self.config.lr, self.config.nb_epochs, self.config.warm_epochs, self.config.power_lr)
            # gamma_cont = self.adjust_loss_pace(epoch, self.config.begin_gamma_cont, self.config.end_gamma_cont, 
            #                                    self.config.nb_epochs, self.config.power_gamma)
            # gamma_clas = self.adjust_loss_pace(epoch, self.config.begin_gamma_clas, self.config.end_gamma_clas, 
            #                                    self.config.nb_epochs, self.config.power_gamma)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Learning Rate: {current_lr}")#, Gamma (cont): {gamma_cont}, Gamma (clas): {gamma_clas}")

            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            train_ense_losses, train_magn_losses, training_loss = 0, 0, 0
            pbar = tqdm(total=nb_batch, desc="Training", mininterval=20)
            for batch in self.loader:
                pbar.update()
                images = batch[0].to(self.device)
                if self.config.text_mode != "none":
                    texts = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                else:
                    texts = None
                    labels = batch[1].to(self.device)
                self.optimizer.zero_grad()

                # with autocast(device_type='cuda' if self.config.cuda else 'cpu'):
                y = self._get_logits(images, texts)
                y = y.float()
                ense_loss = self.ense_loss(y, labels)
                prob = F.softmax(y, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                magn_loss = torch.mean(entropy / np.log(self.config.num_classes))
                batch_loss = (1 - self.lam_mag) * ense_loss + self.lam_mag * magn_loss
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                train_ense_losses += float(ense_loss.item()) / nb_batch
                train_magn_losses += float(magn_loss.item()) / nb_batch
                training_loss += float(batch_loss.item()) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation", mininterval=20)
            val_ense_losses, val_magn_losses, val_loss, correct = 0, 0, 0, 0
            prob_list, label_list, sens_attr_list = [], [], []
            with torch.no_grad():
                self.model.eval()
                for batch in self.loader_val:
                    pbar.update()
                    inputs = batch[0].to(self.device)
                    if self.config.text_mode != "none":
                        texts = batch[1].to(self.device)
                        labels = batch[2].to(self.device)
                        sen_attr = batch[3].to(self.device) if batch[3][0] != -1 else None
                    else:
                        texts = None
                        labels = batch[1].to(self.device)
                        sen_attr = batch[2].to(self.device) if batch[2][0] != -1 else None

                    y = self._get_logits(inputs, texts)
                    # clas_loss_i = selfpaced_weighted(self.clas_loss(y_i, labels), gamma_clas, 'None')
                    # clas_loss_j = selfpaced_weighted(self.clas_loss(y_j, labels), gamma_clas, 'None')
                    ense_loss = self.ense_loss(y, labels)
                    prob = F.softmax(y.float(), dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                    magn_loss = torch.mean(entropy / np.log(self.config.num_classes))
                    batch_loss = (1 - self.lam_mag) * ense_loss + self.lam_mag * magn_loss
                    val_ense_losses += float(ense_loss.item()) / nb_batch
                    val_magn_losses += float(magn_loss.item()) / nb_batch
                    val_loss += float(batch_loss.item()) / nb_batch
                    pred = y.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                    prob_list.append(prob.cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if sen_attr is not None:
                        sens_attr_list.append(sen_attr.cpu().numpy())
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
            
            sum_writer.add_scalar('train/ense_loss', train_ense_losses, epoch)
            sum_writer.add_scalar('train/magn_loss', train_magn_losses, epoch)
            sum_writer.add_scalar('train/loss', training_loss, epoch)
            sum_writer.add_scalar('valid/ense_loss', val_ense_losses, epoch)
            sum_writer.add_scalar('valid/magn_loss', val_magn_losses, epoch)
            sum_writer.add_scalar('valid/loss', val_loss, epoch)
            sum_writer.add_scalar('valid/acc', accuracy, epoch)
            sum_writer.add_scalar('valid/bacc', bacc, epoch)
            sum_writer.add_scalar('valid/f1', f1, epoch)
            if len(sens_attr_list) > 0:
                sum_writer.add_scalar('valid/ES_acc', ES_acc, epoch)
                sum_writer.add_scalar('valid/ES_auc', ES_auc, epoch)
                sum_writer.add_scalar('valid/ES_class_attr_scaled_f1', ES_class_attr_scaled_f1, epoch)


            print("\nEpoch [{}/{}] Train: loss = {:.4f}, Val: loss = {:.4f}, accuracy = {:.4f}, bacc = {:.4f}, f1 = {:.4f}".format(
                epoch, self.config.nb_epochs, training_loss, val_loss, accuracy, bacc, f1), flush=True)
            if len(sens_attr_list) > 0:
                print("Val: ES_acc = {:.4f}, ES_auc = {:.4f}, ES_class_attr_scaled_f1 = {:.4f}".format(ES_acc, ES_auc, ES_class_attr_scaled_f1), flush=True)

            if self.scheduler is not None:
                self.scheduler.step()
            
                
            if ES_class_attr_scaled_f1 is not None and ES_class_attr_scaled_f1 > best_es_class_attr_scaled_f1:
                best_es_class_attr_scaled_f1 = ES_class_attr_scaled_f1
                print("Best model!")
                save_path = os.path.join(self.config.output_dir, "MultiModaCL_Ensemble_Best_fold_{fold}.pth".format(fold=self.fold_id))
                save_trainable_checkpoint(self.model, save_path, epoch)
            elif ES_class_attr_scaled_f1 is None:
                if self.config.metric == "f1" and f1 > best_f1:
                    best_f1 = f1
                    print("Best model!")
                    save_path = os.path.join(self.config.output_dir, "MultiModaCL_Ensemble_Best_fold_{fold}.pth".format(fold=self.fold_id))
                    save_trainable_checkpoint(self.model, save_path, epoch)
                elif self.config.metric == "bacc" and bacc > best_bacc:
                    best_bacc = bacc
                    print("Best model!")
                    save_path = os.path.join(self.config.output_dir, "MultiModaCL_Ensemble_Best_fold_{fold}.pth".format(fold=self.fold_id))
                    save_trainable_checkpoint(self.model, save_path, epoch)
                elif self.config.metric == "acc" and accuracy > best_acc:
                    best_acc = accuracy
                    print("Best model!")
                    save_path = os.path.join(self.config.output_dir, "MultiModaCL_Ensemble_Best_fold_{fold}.pth".format(fold=self.fold_id))
                    save_trainable_checkpoint(self.model, save_path, epoch)
    
    
    def test(self):
        
        # if hasattr(self.config, 'pretrained_path') and self.config.pretrained_path is not None:
        load_trainable_checkpoint(self.model, os.path.join(self.config.output_dir, "MultiModaCL_Ensemble_Best_fold_{fold}.pth".format(fold=self.fold_id)))
        self.model.to(self.device)
        nb_batch = len(self.loader_test)
        pbar = tqdm(total=nb_batch, desc="Test", mininterval=20)
        # predict_dict, truth_dict, count_dict, ensemble_dict = {}, {}, {}, {}
        case_correct = 0
        name_list, predict_list, label_list, prob_list, sens_attr_list = [], [], [], [], []

        with torch.no_grad():
            self.model.eval()
            for batch in self.loader_test:
                pbar.update()
                inputs = batch[0].to(self.device)
                if self.config.text_mode != "none":
                    texts = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    sen_attr = batch[3].to(self.device) if batch[3][0] != -1 else None
                    name = batch[4]
                else:
                    texts = None
                    labels = batch[1].to(self.device)
                    sen_attr = batch[2].to(self.device) if batch[2][0] != -1 else None
                    name = batch[3]

                # patient = name[0].split('/')[0]
                # if patient not in predict_dict:
                #     predict_dict[patient] = 0
                # if patient not in truth_dict:
                #     truth_dict[patient] = labels.cpu().numpy()[0]
                # if patient not in count_dict:
                #     count_dict[patient] = 0
                # count_dict[patient] += 1

                y = self._get_logits(inputs, texts)
                pred = y.argmax(dim=1, keepdim=True)
                prob = F.softmax(y.float(), dim=1)
                case_correct += pred.eq(labels.view_as(pred)).sum().item()
                name_list.append(name[0])
                predict_list.append(torch.flatten(pred).cpu().numpy()[0])
                label_list.append(torch.flatten(labels).cpu().numpy()[0])
                prob_list.append(torch.flatten(prob).cpu().numpy())
                if sen_attr is not None and sen_attr.cpu().numpy()[0] != -1:
                    sens_attr_list.append(sen_attr.cpu().numpy()[0])
        #         predict_dict[patient] += torch.flatten(pred).cpu().numpy()[0]

        # for patient in predict_dict.keys():
        #     ensemble = (predict_dict[patient] / count_dict[patient]) > 0.5
        #     ensemble_dict[patient] = int(ensemble)
        #     if ensemble == truth_dict[patient]:
        #         case_correct += 1

        case_acc = case_correct / len(self.loader_test.dataset)
        # img_acc = img_correct / len(self.loader_test.dataset)
        # case_f1, case_con_matrix, case_class_report = self.metric(list(ensemble_dict.values()), list(truth_dict.values()))
        return_dict = metric(prob_list, label_list, sens_attr_list if len(sens_attr_list) > 0 else None, group_wise=True, equity_alpha=self.config.equity_alpha)
        case_f1 = return_dict['overall_classification_report']['macro avg']['f1-score']
        case_acc = return_dict['overall_classification_report']['accuracy']
        case_auc = return_dict['overall_auc']
        case_bacc = return_dict['overall_bacc']
        case_con_matrix = return_dict['overall_confusion_matrix']
        case_class_report = return_dict['overall_classification_report']
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
        }
        if len(sens_attr_list) > 0:
            csv_dict['case_ES_acc'] = return_dict['overall_ES_acc']
            csv_dict['case_ES_AUC'] = return_dict['overall_ES_AUC']
            csv_dict['case_ES_Class_Attr_Scaled_F1'] = return_dict['overall_ES_Class_Attr_Scaled_F1']
            csv_dict['case_EO'] = return_dict['overall_EO']
            csv_dict['case_DP'] = return_dict['overall_DP']
            csv_dict['case_attr_0_fungal_f1'] = return_dict['attr_0_classification_report']['0']['f1-score']
            csv_dict['case_attr_0_bacterial_f1'] = return_dict['attr_0_classification_report']['1']['f1-score']
            csv_dict['case_attr_0_f1'] = return_dict['attr_0_classification_report']['macro avg']['f1-score']
            csv_dict['case_attr_0_AUC'] = return_dict['attr_0_AUC']
            csv_dict['case_attr_0_acc'] = return_dict['attr_0_classification_report']['accuracy']
            csv_dict['case_attr_1_fungal_f1'] = return_dict['attr_1_classification_report']['0']['f1-score']
            csv_dict['case_attr_1_bacterial_f1'] = return_dict['attr_1_classification_report']['1']['f1-score']
            csv_dict['case_attr_1_f1'] = return_dict['attr_1_classification_report']['macro avg']['f1-score']
            csv_dict['case_attr_1_AUC'] = return_dict['attr_1_AUC']
            csv_dict['case_attr_1_acc'] = return_dict['attr_1_classification_report']['accuracy']
        
        fold_col = f'fold_{self.fold_id}'
        csv_save_path = os.path.join(self.config.output_dir, 'test_results.csv' if self.config.site is None else f'test_results_{self.config.site}.csv')
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
        result = "Case Result: Accuracy = {:.4f}, F1-score = {:.4f}, AUC = {:.4f}, BACC = {:.4f}\n".format(case_acc, case_f1, case_auc, case_bacc)
        if len(sens_attr_list) > 0:
            result += "Equalized Odds (EO) = {:.4f}, Demographic Parity (DP) = {:.4f}\n".format(return_dict['overall_EO'], return_dict['overall_DP'])
            result += "Equalized Accuracy = {:.4f}, Equalized AUC = {:.4f}, Equalized Class-Attr Scaled F1 = {:.4f}\n".format(
                return_dict['overall_ES_acc'], return_dict['overall_ES_AUC'], return_dict['overall_ES_Class_Attr_Scaled_F1'])
        print('\n' + result)
        out_file.write(result)
        case_con_matrix = return_dict['overall_confusion_matrix'].tolist()
        print('Confusion matrix: ')
        out_file.write('Confusion matrix: ' + '\n')
        for l in case_con_matrix:
            print(l)
            out_file.write(str(l) + '\n')
        print('Classification Report: ')
        out_file.write('Classification Report: ' + '\n')
        print(return_dict['overall_classification_report'])
        out_file.write(str(return_dict['overall_classification_report']) + '\n') # type: ignore
        title = 'Name\tprobability\tpredict\ttruth\tsens_attr\n'
        print(title)
        out_file.write(title)
        for i in range(len(name_list)):
            content = str(name_list[i]) + '\t' + str(prob_list[i]) + '\t' + str(predict_list[i]) + '\t' + str(label_list[i]) + '\t' + str(sens_attr_list[i] if len(sens_attr_list) > 0 else -1)
            print(content)
            out_file.write(content + '\n')
        out_file.close()
        
