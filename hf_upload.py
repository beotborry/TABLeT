"""Push the masked-pretraining TABLeT checkpoint to HuggingFace Hub.

Usage:
    huggingface-cli login          # one-time, or set HF_TOKEN
    python hf_upload.py
"""
import os
import sys

import torch
from huggingface_hub import PyTorchModelHubMixin, whoami

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.TABLeT import TABLeT  # noqa: E402

CKPT_PATH = "/mnt/nvme1n1/ours_pretrain/UKB_fixed/best_masked_pretraining_lr0.0001_epoch40_ratio0.5.pth"
REPO_ID = "beotborry/TABLeT_pretrained"
LOCAL_SAVE_DIR = os.path.join(REPO_ROOT, "tablet_pretrained_hf")

# Keys present in the masked-pretraining checkpoint that aren't part of the
# downstream TABLeT encoder. Mirrors src/models/TABLeT.py:171-173, plus
# TR_embedding which the masked-pretraining trainer adds.
PRETRAIN_ONLY_KEYS = ("mask_token", "decoder.weight", "decoder.bias", "TR_embedding")

MODEL_CONFIG = {
    "input_feature_size": 3072,
    "num_tokens": 27 * 256,  # num_tokens_per_frame * num_input_frames
    "num_classes": 1,
    "num_layers": 12,
    "num_heads": 14,
    "num_kv_heads": 2,
    "d_model": 896,
    "d_ff": 4864,
    "dropout": 0.0,
    "feature_dropout": False,
    "hidden_act": "relu",
    "norm_type": "ln",
    "mlp_type": "vanilla",
    "qk_norm": False,
    "input_norm": True,
    "out_norm": False,
    "feature_norm": True,
}


class TABLeTForHub(TABLeT, PyTorchModelHubMixin):
    """`from_pretrained` expands the `config` dict (loaded from config.json by the
    parent mixin) into kwargs for `TABLeT.__init__`, since the 0.20.x mixin doesn't
    do that automatically. User kwargs to `from_pretrained` override config values.
    """

    @classmethod
    def _from_pretrained(cls, *, config=None, **kwargs):
        if config:
            for k, v in config.items():
                kwargs.setdefault(k, v)
        return super()._from_pretrained(**kwargs)


def main():
    try:
        user = whoami()
        print(f"Authenticated as: {user.get('name', user)}")
    except Exception as e:
        sys.exit(
            "Not logged in to HuggingFace. Run `huggingface-cli login` "
            f"or set HF_TOKEN, then retry. ({e!r})"
        )

    print(f"Building TABLeTForHub with config: {MODEL_CONFIG}")
    model = TABLeTForHub(**MODEL_CONFIG)

    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model_state_dict"]
    stripped = [k for k in PRETRAIN_ONLY_KEYS if k in sd]
    for k in stripped:
        sd.pop(k)
    print(f"Stripped pretraining-only keys: {stripped}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"Missing keys after strip: {missing}"
    assert not unexpected, (
        f"Unexpected keys (consider adding to PRETRAIN_ONLY_KEYS): {unexpected}"
    )
    print(
        f"Loaded {sum(p.numel() for p in model.parameters()):,} params "
        f"(epoch={ckpt.get('epoch')}, loss={ckpt.get('loss_value')})"
    )

    print(f"Saving locally to {LOCAL_SAVE_DIR}")
    model.save_pretrained(LOCAL_SAVE_DIR, config=MODEL_CONFIG)

    commit_msg = (
        f"Upload masked-pretraining checkpoint "
        f"(epoch={ckpt.get('epoch')}, loss={ckpt.get('loss_value'):.4f})"
    )
    print(f"Pushing to https://huggingface.co/{REPO_ID}")
    model.push_to_hub(REPO_ID, config=MODEL_CONFIG, private=False, commit_message=commit_msg)

    print("Round-trip verifying via from_pretrained...")
    reloaded = TABLeTForHub.from_pretrained(REPO_ID)
    orig_sd = model.state_dict()
    new_sd = reloaded.state_dict()
    assert orig_sd.keys() == new_sd.keys(), "Key set differs after reload"
    for k in orig_sd:
        assert torch.equal(orig_sd[k], new_sd[k]), f"Tensor mismatch: {k}"
    print(
        f"Round-trip OK. Reloaded {sum(p.numel() for p in reloaded.parameters()):,} params."
    )
    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
