from src.trainers.generic_trainer import GenericTrainer
from src.trainers.utils import Percept_Loss
import torch.nn as nn
from src.trainers.utils import get_intense_voxels
import wandb
import torch
import monai.transforms as monai_t
from src.trainers.swift_losses import NTXentLoss

class mlp(nn.Module):

    def __init__(self, final_embedding_size=128, num_tokens = 196, use_normalization=True, n_local_frames=4):

        super(mlp, self).__init__()

        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(num_tokens, self.final_embedding_size, bias=False)
        self.bn1 = nn.BatchNorm1d(self.final_embedding_size)
        self.temp_avg = nn.AdaptiveAvgPool1d(1)  #
        self.n_local_frames = n_local_frames

    def forward(self, x, type):
        # x -> b, 96, 4, 4, 4, t
        
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C

        if type == 'l':
            # Global Dense Representation, this will be used for IC and LL losses
            x = self.temp_avg(x.transpose(1, 2))
            x = x.flatten(1)
            x = nn.functional.normalize(self.bn1(self.fc1(x)), p=2, dim=1)
            return x

        elif type == 'g':
            # Global Sparse Representation, this will be used for the IC loss
            gsr = self.temp_avg(x.transpose(1, 2))
            gsr = gsr.flatten(1)
            gsr = nn.functional.normalize(self.bn1(self.fc1(gsr)), p=2, dim=1)
            return gsr

        else:
            return None, None, None, None, None

class SwiftPretrainingTrainer(GenericTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, config, args, accelerator, logger, run):
        super().__init__(model, train_loader, val_loader, test_loader, optimizer, scheduler, config, args, accelerator, logger, run)
        self.scheduler = scheduler
        self.best_val_loss = 100000000
        self.output_head = mlp()
        
    def augment(self, img):
        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            # 0.175 rad = 10 degrees
            rotate_range=(0.175, 0.175, 0.175),
            scale_range = (0.1, 0.1, 0.1),
            mode = "bilinear",
            padding_mode = "border",
            device = device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            # set augmentation seed to be the same for all time steps
            for t in range(T):
                if self.hparams.augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')

        return img
        
    
    def train_epoch(self, epoch, epochs, verbose, save_dir):
        self.model.train()
        total_loss = 0.0
        total_ic_loss = 0.0
        total_ll_loss = 0.0
        total_batches = len(self.train_loader)

        for batch_idx, inputs in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img, label, (subject, TR, attn_mask) = inputs
            img, diff_img = img
            img = img.float()
            diff_img = diff_img.float()
            
            loss = 0
            batch_size = img.shape[0]
            criterion = NTXentLoss(device='cuda', batch_size=batch_size, temperature=0.1, use_cosine_similarity=True).cuda()
            criterion_l1 = NTXentLoss(device='cuda', batch_size=2, temperature=0.1, use_cosine_similarity=True).cuda()
            
            out_global_1 = self.output_head(self.model(self.augment(img), get_inter=True), "g")
            out_global_2 = self.output_head(self.model(self.augment(diff_img), get_inter=True), "g")
            ic_loss = criterion(out_global_1, out_global_2)
            loss += ic_loss
            total_ic_loss += ic_loss.item()
            
            out_local_1 = []
            out_local_2 = []
            out_local_swin1 = self.model(self.augment(img), get_inter=True)
            out_local_swin2 = self.model(self.augment(img), get_inter=True)
            out_local_1.append(self.output_head(out_local_swin1, "l"))
            out_local_2.append(self.output_head(out_local_swin2, "l"))
            
            out_local_swin1 = self.model(self.augment(diff_img), get_inter=True)
            out_local_swin2 = self.model(self.augment(diff_img), get_inter=True)
            out_local_1.append(self.output_head(out_local_swin1,"l"))
            out_local_2.append(self.output_head(out_local_swin2,"l"))

            ll_loss = 0
            for i in range(out_local_1[0].shape[0]):
                ll_loss += criterion_l1(torch.stack(out_local_1, dim=1)[i], torch.stack(out_local_2, dim=1)[i])
            total_ll_loss += ll_loss.item()
            loss += ll_loss
            
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_ic_loss += ic_loss.item()
            total_ll_loss += ll_loss.item()
        avg_loss = total_loss / total_batches
        avg_ic_loss = total_ic_loss / total_batches
        avg_ll_loss = total_ll_loss / total_batches


        print('Train epoch: {}/{} | Loss: {:.4f} | IC Loss: {:.4f} | LL Loss: {:.4f}'.format(
            epoch + 1, epochs, avg_loss, avg_ic_loss, avg_ll_loss))

        wandb.log({
            'train/loss': avg_loss,
            'train/ic_loss': avg_ic_loss,
            'train/ll_loss': avg_ll_loss
        }, step=epoch)
        
        
    def val_epoch(self, epoch, epochs):
        self.model.eval()
        total_loss = 0.0
        total_ic_loss = 0.0
        total_ll_loss = 0.0
        total_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                img, label, (subject, TR, attn_mask) = inputs
                img, diff_img = img
                img = img.float()
                diff_img = diff_img.float()
                loss = 0
                batch_size = img.shape[0]
                criterion = NTXentLoss(device='cuda', batch_size=batch_size, temperature=0.1, use_cosine_similarity=True).cuda()
                criterion_l1 = NTXentLoss(device='cuda', batch_size=2, temperature=0.1, use_cosine_similarity=True).cuda()
                
                out_global_1 = self.output_head(self.model(self.augment(img), get_inter=True), "g")
                out_global_2 = self.output_head(self.model(self.augment(diff_img), get_inter=True), "g")

                ic_loss = criterion(out_global_1, out_global_2)
                loss += ic_loss
                total_ic_loss += ic_loss.item()
                out_local_1 = []
                out_local_2 = []
                out_local_swin1 = self.model(self.augment(img), get_inter=True)
                out_local_swin2 = self.model(self.augment(img), get_inter=True)
                out_local_1.append(self.output_head(out_local_swin1, "l"))
                out_local_2.append(self.output_head(out_local_swin2, "l"))

                out_local_swin1 = self.model(self.augment(diff_img), get_inter=True)
                out_local_swin2 = self.model(self.augment(diff_img), get_inter=True)
                out_local_1.append(self.output_head(out_local_swin1,"l"))
                out_local_2.append(self.output_head(out_local_swin2,"l"))

                ll_loss = 0
                for i in range(out_local_1[0].shape[0]):
                    ll_loss += criterion_l1(torch.stack(out_local_1, dim=1)[i], torch.stack(out_local_2, dim=1)[i])
                total_ll_loss += ll_loss.item()
                loss += ll_loss
                total_loss += loss.item()

            avg_loss = total_loss / total_batches
            avg_ic_loss = total_ic_loss / total_batches
            avg_ll_loss = total_ll_loss / total_batches

            print('Val epoch: {}/{} | Loss: {:.4f} | IC Loss: {:.4f} | LL Loss: {:.4f}'.format(
                epoch + 1, epochs, avg_loss, avg_ic_loss, avg_ll_loss))

            
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self.save_checkpoint(epoch)
            
            wandb.log({
                'val/loss': avg_loss,
                'val/ic_loss': avg_ic_loss,
                'val/ll_loss': avg_ll_loss,
                'val/best_loss': self.best_val_loss
            }, step=epoch)
    
    def save_checkpoint(self, epoch):
        self.model.save_checkpoint(
            self.config.train.save_dir,
            f"swift_pretraining_lr{self.args.lr}_epoch{self.args.epochs}_bs{self.args.batch_size}",
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