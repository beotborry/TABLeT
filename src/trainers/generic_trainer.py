import os

import torch
import torch.nn as nn
import numpy as np
import wandb
import sklearn.metrics
import scipy.stats

CLASSIFICATION_TASK = ["gender", "CNorMCI", "ADHD", "Autism", "TCP", "MovieClf"]
REGRESSION_TASK = ["age", "intelligence"]


class GenericTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler,
                 config, args, accelerator, logger, run):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.args = args
        self.accelerator = accelerator
        self.logger = logger
        self.task = config.data.task
        self.run = run

        self.val_best_epoch = {}
        self.val_best_metric = {}
        self.test_metrics = {}

        if config.data.task in REGRESSION_TASK:
            self.criterion = nn.L1Loss()
            self.main_metrics = ['pearson', 'mae', 'mse']
        elif config.data.task in CLASSIFICATION_TASK:
            self.main_metrics = ['f1', 'accuracy', 'auroc', 'loss', 'avg_loss']
            if config.data.pos_weight:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(train_loader.dataset.pos_weight).to(accelerator.device)
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()

        if config.data.task in REGRESSION_TASK or config.data.task in CLASSIFICATION_TASK:
            for metric in self.main_metrics:
                self.val_best_epoch[metric] = 0
                self.val_best_metric[metric] = np.inf if metric in ['mae', 'mse', 'loss', 'avg_loss'] else -np.inf
                self.test_metrics[metric] = []

    def _forward(self, img, TR, attn_mask):
        if self.config.train.use_TR:
            return self.model(img, TR=TR, attn_mask=attn_mask)
        return self.model(img, attn_mask=attn_mask)

    def train_epoch(self, epoch, epochs, verbose, save_dir):
        self.model.train()
        total_loss = 0.0
        total_batches = len(self.train_loader)
        preds = []
        labels = []
        total_correct = 0

        for batch_idx, inputs in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img, label, (subject, TR, attn_mask) = inputs
            label = label.type(img.dtype)
            outputs = self._forward(img, TR, attn_mask)
            if len(label) == 1 and label.shape != outputs.shape:
                outputs = outputs.unsqueeze(0)
            loss = self.criterion(outputs, label)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if self.task in CLASSIFICATION_TASK:
                pred = torch.sigmoid(outputs)
                preds.append(pred.detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())
                total_correct += (pred.round() == label).sum().item()
            elif self.task in REGRESSION_TASK:
                preds.append(outputs.detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

        train_loss = total_loss / total_batches
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        if self.accelerator.is_main_process:
            log_dict = {"train/loss": train_loss, "lr": self.scheduler.get_last_lr()[0]}

            if self.task in CLASSIFICATION_TASK:
                accuracy = total_correct / len(self.train_loader.dataset)
                auroc = sklearn.metrics.roc_auc_score(labels, preds)
                f1 = sklearn.metrics.f1_score(labels, preds.round())
                sensitivity = sklearn.metrics.recall_score(labels, preds.round())
                specificity = sklearn.metrics.recall_score(labels, preds.round(), pos_label=0)
                log_dict.update({
                    "train/accuracy": accuracy, "train/auroc": auroc, "train/f1": f1,
                    "train/sensitivity": sensitivity, "train/specificity": specificity,
                })
                self.logger.info(f"[TRAIN] Epoch [{epoch + 1:02d}/{epochs}] - Accuracy: {accuracy:.4f} - AUROC: {auroc:.4f} - F1: {f1:.4f}")
            elif self.task in REGRESSION_TASK:
                mse = sklearn.metrics.mean_squared_error(labels, preds)
                mae = sklearn.metrics.mean_absolute_error(labels, preds)
                pearson = scipy.stats.pearsonr(labels, preds)[0]
                log_dict.update({"train/mse": mse, "train/mae": mae, "train/pearson": pearson})
                self.logger.info(f"[TRAIN] Epoch [{epoch + 1:02d}/{epochs}] - MSE: {mse:.4f} - MAE: {mae:.4f} - Pearson: {pearson:.4f}")

            wandb.log(log_dict, step=epoch)
            self.logger.info(f"[TRAIN] Epoch [{epoch + 1:02d}/{epochs}] - Loss: {train_loss:.4f}")
            os.makedirs(save_dir, exist_ok=True)

    def _aggregate_by_subject(self, loader):
        """Run inference and aggregate logits/labels by subject."""
        self.model.eval()
        total_loss = 0.0
        total_batches = len(loader)
        logits_by_subject = {}
        labels_by_subject = {}

        with torch.no_grad():
            for inputs in loader:
                img, label, (subject, TR, attn_mask) = inputs
                label = label.type(img.dtype)
                outputs = self._forward(img, TR, attn_mask)
                if len(label) == 1 and label.shape != outputs.shape:
                    outputs = outputs.unsqueeze(0)
                loss = self.criterion(outputs, label)

                logit = outputs.detach().cpu().numpy()
                label_np = label.detach().cpu().numpy()
                total_loss += loss.item()

                for i, sub in enumerate(subject):
                    if sub not in logits_by_subject:
                        logits_by_subject[sub] = []
                        labels_by_subject[sub] = []
                        labels_by_subject[sub].append(label_np[i])
                    logits_by_subject[sub].append(logit[i])

            for sub in logits_by_subject:
                logits_by_subject[sub] = np.array(logits_by_subject[sub]).mean(axis=0)
                labels_by_subject[sub] = np.array(labels_by_subject[sub])

        # Sort by subject for reproducibility
        logits_by_subject = dict(sorted(logits_by_subject.items()))
        labels_by_subject = dict(sorted(labels_by_subject.items()))
        assert logits_by_subject.keys() == labels_by_subject.keys()

        avg_loss = total_loss / total_batches
        all_logits = np.array(list(logits_by_subject.values()))
        all_labels = np.array(list(labels_by_subject.values())).squeeze(axis=1)
        total_avg_loss = self.criterion(
            torch.tensor(all_logits).cuda(), torch.tensor(all_labels).cuda()
        ).item()

        if self.task in CLASSIFICATION_TASK:
            preds = torch.sigmoid(torch.tensor(all_logits)).numpy()
        else:
            preds = all_logits

        return avg_loss, total_avg_loss, preds, all_labels, logits_by_subject

    def val_epoch(self, epoch, epochs):
        val_loss, total_avg_loss, preds, labels, logits_by_subject = self._aggregate_by_subject(self.val_loader)

        if self.task in CLASSIFICATION_TASK:
            if self.args.logit_save_dir is not None:
                os.makedirs(self.args.logit_save_dir, exist_ok=True)
                np.save(os.path.join(self.args.logit_save_dir, f"val_logits_epoch{epoch+1:02d}.npy"), preds)
                np.save(os.path.join(self.args.logit_save_dir, f"val_labels_epoch{epoch+1:02d}.npy"), labels)

            total_correct = (preds.round() == labels).sum()
            val_accuracy = total_correct / len(logits_by_subject)
            val_auroc = sklearn.metrics.roc_auc_score(labels, preds)
            val_f1 = sklearn.metrics.f1_score(labels, preds.round())
            val_sensitivity = sklearn.metrics.recall_score(labels, preds.round())
            val_specificity = sklearn.metrics.recall_score(labels, preds.round(), pos_label=0)

            for metric, val in [('f1', val_f1), ('accuracy', val_accuracy), ('auroc', val_auroc)]:
                if self.val_best_metric[metric] < val:
                    self.val_best_metric[metric] = val
                    self.val_best_epoch[metric] = epoch
            for metric, val in [('loss', val_loss), ('avg_loss', total_avg_loss)]:
                if self.val_best_metric[metric] > val:
                    self.val_best_metric[metric] = val
                    self.val_best_epoch[metric] = epoch

        elif self.task in REGRESSION_TASK:
            val_mse = sklearn.metrics.mean_squared_error(labels, preds)
            val_mae = sklearn.metrics.mean_absolute_error(labels, preds)
            val_pearson = scipy.stats.pearsonr(labels, preds)[0]

            for metric, val in [('mae', val_mae), ('mse', val_mse)]:
                if self.val_best_metric[metric] > val:
                    self.val_best_metric[metric] = val
                    self.val_best_epoch[metric] = epoch
            if self.val_best_metric['pearson'] < val_pearson:
                self.val_best_metric['pearson'] = val_pearson
                self.val_best_epoch['pearson'] = epoch

        if self.accelerator.is_main_process:
            log_dict = {"val/loss": val_loss, "val/avg_loss": total_avg_loss}
            if self.task in CLASSIFICATION_TASK:
                log_dict.update({
                    "val/accuracy": val_accuracy, "val/auroc": val_auroc, "val/f1": val_f1,
                    "val/sensitivity": val_sensitivity, "val/specificity": val_specificity,
                })
                self.logger.info(f"[VAL] Epoch [{epoch + 1:02d}/{epochs}] - Accuracy: {val_accuracy:.4f} - AUROC: {val_auroc:.4f} - F1: {val_f1:.4f}")
            elif self.task in REGRESSION_TASK:
                log_dict.update({"val/mse": val_mse, "val/mae": val_mae, "val/pearson": val_pearson})
                self.logger.info(f"[VAL] Epoch [{epoch + 1:02d}/{epochs}] - MSE: {val_mse:.4f} - MAE: {val_mae:.4f} - Pearson: {val_pearson:.4f}")
            wandb.log(log_dict, step=epoch)

    def test(self, epoch, epochs):
        test_loss, total_avg_loss, preds, labels, _ = self._aggregate_by_subject(self.test_loader)

        if self.task in CLASSIFICATION_TASK:
            total_correct = (preds.round() == labels).sum()
            test_accuracy = total_correct / len(labels)
            test_auroc = sklearn.metrics.roc_auc_score(labels, preds)
            test_f1 = sklearn.metrics.f1_score(labels, preds.round())
            test_sensitivity = sklearn.metrics.recall_score(labels, preds.round())
            test_specificity = sklearn.metrics.recall_score(labels, preds.round(), pos_label=0)

            self.test_metrics['f1'].append(test_f1)
            self.test_metrics['accuracy'].append(test_accuracy)
            self.test_metrics['auroc'].append(test_auroc)
            self.test_metrics['loss'].append(test_loss)
            self.test_metrics['avg_loss'].append(total_avg_loss)
        elif self.task in REGRESSION_TASK:
            test_mse = sklearn.metrics.mean_squared_error(labels, preds)
            test_mae = sklearn.metrics.mean_absolute_error(labels, preds)
            test_pearson = scipy.stats.pearsonr(labels, preds)[0]

            self.test_metrics['mae'].append(test_mae)
            self.test_metrics['mse'].append(test_mse)
            self.test_metrics['pearson'].append(test_pearson)

        if self.accelerator.is_main_process:
            log_dict = {"test/loss": test_loss, "test/avg_loss": total_avg_loss}
            self.logger.info(f"[TEST] Test Loss: {test_loss:.4f}")
            if self.task in CLASSIFICATION_TASK:
                log_dict.update({
                    "test/accuracy": test_accuracy, "test/auroc": test_auroc, "test/f1": test_f1,
                    "test/sensitivity": test_sensitivity, "test/specificity": test_specificity,
                })
                self.logger.info(f"[TEST] Accuracy: {test_accuracy:.4f} - AUROC: {test_auroc:.4f} - F1: {test_f1:.4f}")
            elif self.task in REGRESSION_TASK:
                log_dict.update({"test/mse": test_mse, "test/mae": test_mae, "test/pearson": test_pearson})
                self.logger.info(f"[TEST] MSE: {test_mse:.4f} - MAE: {test_mae:.4f} - Pearson: {test_pearson:.4f}")
            wandb.log(log_dict, step=epoch)

    def report_best_test_metrics(self):
        if self.accelerator.is_main_process:
            self.logger.info("[TEST] Best Test Metrics:")
            wandb_log_info = {}

            for val_metric in self.main_metrics:
                for test_metric in self.main_metrics:
                    best_epoch = self.val_best_epoch[val_metric]
                    val = self.test_metrics[test_metric][best_epoch]
                    self.logger.info(f"[TEST] Best {test_metric}: {val:.4f} at epoch {best_epoch} (by val {val_metric})")
                    wandb_log_info[f"test/best_{test_metric}_from_val_{val_metric}"] = val

            for val_metric in self.main_metrics:
                self.logger.info(f"[TEST] Best Val {val_metric}: {self.val_best_metric[val_metric]:.4f} at epoch {self.val_best_epoch[val_metric]}")
                wandb_log_info[f"val/best_{val_metric}"] = self.val_best_metric[val_metric]
                wandb_log_info[f"val/best_epoch_{val_metric}"] = self.val_best_epoch[val_metric]

            wandb.log(wandb_log_info)
