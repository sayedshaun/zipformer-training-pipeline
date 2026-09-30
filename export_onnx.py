"""Export a trained RNNT checkpoint to three ONNX graphs: encoder, decoder
(prediction-network single step) and joiner. RNNT can't export as one graph -
the prediction network is autoregressive, so a streaming/step-by-step runtime
(e.g. sherpa-onnx, or a hand-rolled onnxruntime loop) calls these three in the
same pattern as ZipformerFromScratch.greedy_rnnt: run the encoder once, then
per frame call decoder+joiner in a loop, advancing the decoder state only on
a non-blank emission.

Usage:
    python export_onnx.py --config config.yaml \
        --model zipformer_bn/rnnt_final_epoch28.pt \
        --out onnx_export
"""

import argparse
from pathlib import Path

import torch

from src.config import load_sections
from src.model import ZipformerFromScratch
from tokenizer import BPETokenizer


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="onnx_export")
    parser.add_argument("--opset", type=int, default=17)
    return parser


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform, waveform_lengths):
        return self.model.encode(waveform, waveform_lengths)


class DecoderStepWrapper(torch.nn.Module):
    """One prediction-network step: token in, LSTM state in -> pred_out, state out."""

    def __init__(self, pred_net):
        super().__init__()
        self.pred_net = pred_net

    def forward(self, token, h0, c0):
        pred_out, (h1, c1) = self.pred_net.step(token, (h0, c0))
        return pred_out, h1, c1


class JoinerStepWrapper(torch.nn.Module):
    """One (frame, decoder-output) pair -> logits, matching greedy_rnnt's per-step joint."""

    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, encoder_frame, pred_out):
        enc = self.joint.enc_proj(encoder_frame)
        pred = self.joint.pred_proj(pred_out)
        return self.joint.out(self.joint.activation(enc + pred))


def main():
    args = build_arg_parser().parse_args()
    manifests_args = load_sections(args.config, "manifests")
    model_args = load_sections(args.config, "model")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.load(str(Path(manifests_args.output_dir) / "tokenizer.model"))
    model = ZipformerFromScratch(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args.d_model,
        n_layers=model_args.n_layers,
        n_heads=model_args.n_heads,
        attn_head_dim=model_args.attn_head_dim,
        nla_hidden_dim=model_args.nla_hidden_dim,
        conv_kernel_size=model_args.conv_kernel_size,
        ff_expansion_factor=model_args.ff_expansion_factor,
        dropout=model_args.dropout,
        bypass_min_scale=model_args.bypass_min_scale,
        use_rnnt=True,
    )
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded {args.model} (epoch {ckpt.get('epoch')}, step {ckpt.get('step')})")

    pred_dim = model.prediction_network.lstm.hidden_size
    pred_layers = model.prediction_network.lstm.num_layers
    d_model = model.encoder.d_model

    # --- encoder ---
    sample_waveform = torch.randn(1, 16000 * 3)  # 3s dummy clip
    sample_lengths = torch.tensor([16000 * 3], dtype=torch.long)
    encoder_wrapper = EncoderWrapper(model)
    torch.onnx.export(
        encoder_wrapper,
        (sample_waveform, sample_lengths),
        str(out_dir / "encoder.onnx"),
        input_names=["waveform", "waveform_lengths"],
        output_names=["encoder_out", "encoded_lengths"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "waveform_lengths": {0: "batch"},
            "encoder_out": {0: "batch", 1: "frames"},
            "encoded_lengths": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"Exported {out_dir / 'encoder.onnx'}")

    # --- decoder (prediction network, one step) ---
    decoder_wrapper = DecoderStepWrapper(model.prediction_network)
    sample_token = torch.full((1, 1), model.prediction_network.blank_id, dtype=torch.long)
    sample_h0 = torch.zeros(pred_layers, 1, pred_dim)
    sample_c0 = torch.zeros(pred_layers, 1, pred_dim)
    torch.onnx.export(
        decoder_wrapper,
        (sample_token, sample_h0, sample_c0),
        str(out_dir / "decoder.onnx"),
        input_names=["token", "h0", "c0"],
        output_names=["pred_out", "h1", "c1"],
        dynamic_axes={
            "token": {0: "batch"},
            "h0": {1: "batch"},
            "c0": {1: "batch"},
            "pred_out": {0: "batch"},
            "h1": {1: "batch"},
            "c1": {1: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"Exported {out_dir / 'decoder.onnx'}")

    # --- joiner ---
    joiner_wrapper = JoinerStepWrapper(model.joint_network)
    sample_encoder_frame = torch.randn(1, d_model)
    sample_pred_out = torch.randn(1, pred_dim)
    torch.onnx.export(
        joiner_wrapper,
        (sample_encoder_frame, sample_pred_out),
        str(out_dir / "joiner.onnx"),
        input_names=["encoder_frame", "pred_out"],
        output_names=["logits"],
        dynamic_axes={
            "encoder_frame": {0: "batch"},
            "pred_out": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"Exported {out_dir / 'joiner.onnx'}")

    import shutil
    shutil.copy(str(Path(manifests_args.output_dir) / "tokenizer.model"), str(out_dir / "tokenizer.model"))
    print(f"Copied tokenizer.model -> {out_dir / 'tokenizer.model'}")
    print(f"blank_id={model.prediction_network.blank_id} pred_dim={pred_dim} pred_layers={pred_layers} d_model={d_model}")


if __name__ == "__main__":
    main()
