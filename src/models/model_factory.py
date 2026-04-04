import torch
import ast


class ModelFactory:
    def __init__(self, config, args):
        self.config = config
        self.args = args

    def create_model(self):
        torch.manual_seed(self.args.seed)
        if self.config.model.name == "TABLeT":
            from src.models.TABLeT import TABLeT
            model = TABLeT(
                input_feature_size=self.config.model.params.input_feature_size,
                num_tokens=self.config.model.params.num_tokens_per_frame * self.config.data.num_input_frames,
                num_classes=self.config.model.params.num_classes,
                num_layers=self.config.model.params.num_layers,
                num_heads=self.config.model.params.num_heads,
                num_kv_heads=self.config.model.params.num_kv_heads,
                d_model=self.config.model.params.d_model,
                d_ff=self.config.model.params.d_ff,
                dropout=self.config.model.params.dropout,
                feature_dropout=self.config.model.params.feature_dropout,
                hidden_act=self.config.model.params.hidden_act,
                norm_type=self.config.model.params.norm_type,
                mlp_type=self.config.model.params.mlp_type if 'mlp_type' in self.config.model.params else 'vanilla',
                qk_norm=self.config.model.params.qk_norm,
                input_norm=self.config.model.params.input_norm,
                out_norm=self.config.model.params.out_norm,
                feature_norm=self.config.model.params.feature_norm,
                tablet_pretrained_weight_path=self.args.tablet_pretrained_weight_path
            )
        elif self.config.model.name == "Swift":
            from src.models.swift import SwinTransformer4D
            model = SwinTransformer4D(
                img_size=(96, 96, 96, self.config.data.num_input_frames),
                in_chans=1,
                embed_dim=36,
                window_size=self.config.model.params.window_size,
                first_window_size=self.config.model.params.window_size,
                patch_size=(6, 6, 6, 1),
                depths=self.config.model.params.depths,
                num_heads=self.config.model.params.num_heads,
                downsample="mergingv2",
                to_float=True,
                last_layer_full_MSA=True,
            )
        elif self.config.model.name == "BrainNetCNN":
            from src.models.brain_net_cnn import BrainNetCNN
            model = BrainNetCNN(**self.config.model.params)
        elif self.config.model.name == "BrainNetworkTransformer":
            from src.models.brain_net_transformer import BrainNetworkTransformer
            model = BrainNetworkTransformer(self.config.model.params)
        elif self.config.model.name == "MeanMLP":
            from src.models.mean_mlp import meanMLP
            model = meanMLP()
        elif self.config.model.name == "AutoEncoder":
            from src.models.tff import AutoEncoder
            model = AutoEncoder(dim=(96, 96, 96), **self.config.model.params)
        elif self.config.model.name == "Encoder_Transformer_Decoder":
            from src.models.tff import Encoder_Transformer_Decoder
            model = Encoder_Transformer_Decoder(dim=(96, 96, 96), **self.config.model.params)
        elif self.config.model.name == "Encoder_Transformer_finetune":
            from src.models.tff import Encoder_Transformer_finetune
            from src.trainers.generic_trainer import CLASSIFICATION_TASK, REGRESSION_TASK

            if self.config.data.task in CLASSIFICATION_TASK:
                task = 'binary_classification'
            elif self.config.data.task in REGRESSION_TASK:
                task = 'regression'

            model = Encoder_Transformer_finetune(dim=(96, 96, 96), fine_tune_task=task, **self.config.model.params)
        elif self.config.model.name == "BrainJEPA":
            from src.models.brain_jepa import VisionTransformer
            model = VisionTransformer(
                patch_size=self.config.model.params.patch_size,
                crop_size=ast.literal_eval(self.config.model.params.crop_size),
                pred_depth=self.config.model.params.pred_depth,
                pred_emb_dim=self.config.model.params.pred_emb_dim,
                add_w=self.config.model.params.add_w,
                gradient_checkpointing=self.config.model.params.gradient_checkpointing,
                model_name='vit_base',
                attn_mode='flash_attn',
                global_pool=True,
                device='cuda',
                num_classes=1,
                gradient_csv_path=f'{self.config.data.dataset}_tid{self.args.test_set_id}_gradients.pt'
            )
        elif self.config.model.name == "BrainJEPA_finetune":
            from src.models.brain_jepa import VisionTransformer
            model = VisionTransformer(
                patch_size=self.config.model.params.patch_size,
                crop_size=ast.literal_eval(self.config.model.params.crop_size),
                pred_depth=self.config.model.params.pred_depth,
                pred_emb_dim=self.config.model.params.pred_emb_dim,
                add_w=self.config.model.params.add_w,
                gradient_checkpointing=self.config.model.params.gradient_checkpointing,
                model_name='vit_base',
                attn_mode='flash_attn',
                global_pool=True,
                device='cuda',
                num_classes=1,
                gradient_csv_path=self.config.model.params.gradient_csv_path
            )
            checkpoint_path = self.config.model.params.pretrained_checkpoint_path
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % checkpoint_path)
            checkpoint_model = checkpoint['target_encoder']
            state_dict = model.state_dict()

            new_checkpoint_model = {}
            for key in checkpoint_model.keys():
                new_key = key.replace('module.', 'encoder.')
                new_checkpoint_model[new_key] = checkpoint_model[key]

            for k in ['head.weight', 'head.bias']:
                if k in new_checkpoint_model and new_checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del new_checkpoint_model[k]

            msg = model.load_state_dict(new_checkpoint_model, strict=False)
            print(msg)

            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

            from timm.models.layers import trunc_normal_
            trunc_normal_(model.head.weight, std=2e-5)

        else:
            raise ValueError(f"Unsupported model name: {self.config.model.name}")

        return model
