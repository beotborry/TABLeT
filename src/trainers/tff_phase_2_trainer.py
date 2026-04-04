from src.trainers.generic_trainer import GenericTrainer
from src.trainers.utils import Percept_Loss
import torch.nn as nn
from src.trainers.utils import get_intense_voxels
import wandb
import torch

class TFFPhase2Trainer(GenericTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, config, args, accelerator, logger, run, phase1_weight_path=None):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, config, args, accelerator, logger, run)
        self.recon_cri = nn.L1Loss()
        self.intensity_cri = nn.L1Loss()
        self.percept_cri = Percept_Loss(task='autoencoder_reconstruction', cuda=True)
        self.scheduler = scheduler
        self.best_val_loss = 100000000
        
        assert phase1_weight_path is not None
        state_dict = torch.load(phase1_weight_path)
        self.model.load_partial_state_dict(state_dict['model_state_dict'], load_cls_embedding=False)
        self.model.loaded_model_weights_path = phase1_weight_path

    def compute_reconstruction(self,recon,img):
        fmri_sequence = img[:,0].unsqueeze(1)
        reconstruction_loss = self.recon_cri(recon,fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,recon,img):
        per_voxel = img[:,1,:,:,:,:]
        voxels = get_intense_voxels(per_voxel, recon.shape)
        output_intense = recon[voxels]
        truth_intense = img[:,0][voxels.squeeze(1)]
        intensity_loss = self.intensity_cri(output_intense.squeeze(), truth_intense)
        return intensity_loss

    def compute_perceptual(self,recon,img):
        fmri_sequence = img[:,0].unsqueeze(1)
        perceptual_loss = self.percept_cri(recon,fmri_sequence)
        return perceptual_loss
    
    def train_epoch(self, epoch, epochs, verbose, save_dir):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_intensity_loss = 0.0
        total_perceptual_loss = 0.0
        total_batches = len(self.train_loader)

        for batch_idx, inputs in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img, label, (subject, TR, attn_mask) = inputs
            img = img.float()
            outputs = self.model(img)['reconstructed_fmri_sequence'] # (B, C, W, H, D, T)

            recon_loss = self.compute_reconstruction(outputs, img)
            intensity_loss = self.compute_intensity(outputs, img)
            perceptual_loss = self.compute_perceptual(outputs, img)

            loss = recon_loss + intensity_loss + perceptual_loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_intensity_loss += intensity_loss.item()
            total_perceptual_loss += perceptual_loss.item()

        avg_loss = total_loss / total_batches
        avg_recon_loss = total_recon_loss / total_batches
        avg_intensity_loss = total_intensity_loss / total_batches
        avg_perceptual_loss = total_perceptual_loss / total_batches

        print('Train epoch: {}/{} | Loss: {:.4f} | Recon: {:.4f} | Intensity: {:.4f} | Perceptual: {:.4f}'.format(
            epoch + 1, epochs, avg_loss, avg_recon_loss, avg_intensity_loss, avg_perceptual_loss))

        wandb.log({
            'train/loss': avg_loss,
            'train/recon_loss': avg_recon_loss,
            'train/intensity_loss': avg_intensity_loss,
            'train/perceptual_loss': avg_perceptual_loss
        }, step=epoch)
        
        
    def val_epoch(self, epoch, epochs):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_intensity_loss = 0.0
        total_perceptual_loss = 0.0
        total_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                img, label, (subject, TR, attn_mask) = inputs
                img = img.float()
                outputs = self.model(img)['reconstructed_fmri_sequence'] # (B, C, W, H, D, T)

                recon_loss = self.compute_reconstruction(outputs, img)
                intensity_loss = self.compute_intensity(outputs, img)
                perceptual_loss = self.compute_perceptual(outputs, img)

                loss = recon_loss + intensity_loss + perceptual_loss
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_intensity_loss += intensity_loss.item()
                total_perceptual_loss += perceptual_loss.item()

            avg_loss = total_loss / total_batches
            avg_recon_loss = total_recon_loss / total_batches
            avg_intensity_loss = total_intensity_loss / total_batches
            avg_perceptual_loss = total_perceptual_loss / total_batches

            print('Val epoch: {}/{} | Loss: {:.4f} | Recon: {:.4f} | Intensity: {:.4f} | Perceptual: {:.4f}'.format(
                epoch + 1, epochs, avg_loss, avg_recon_loss, avg_intensity_loss, avg_perceptual_loss))

            wandb.log({
                'val/loss': avg_loss,
                'val/recon_loss': avg_recon_loss,
                'val/intensity_loss': avg_intensity_loss,
                'val/perceptual_loss': avg_perceptual_loss
            }, step=epoch)

            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        self.model.save_checkpoint(
            self.config.train.save_dir,
            f"tff_phase2_tid{self.args.test_set_id}_lr{self.args.lr}_epoch{self.args.epochs}_bs{self.args.batch_size}",
            epoch,
            self.best_val_loss,
            accuracy=None,
            optimizer=None,
            schedule=None,
        )

    def test(self, epoch, epochs):
        return
    
    def report_best_test_metrics(self):
        return