import argparse

import torch
from torch import nn

from ofa.configuration_ofa import OFAConfig
from ofa.modeling_ofa import OFAModel
from architecture_configs import ofa_base, ofa_large, ofa_tiny


def trans_fairseq_to_huggingface(fs_model, hf_model, config):
    model = torch.load(fs_model, map_location='cpu')
    state = model["model"]
    keys = list(state.keys())
    for k in keys:
        if 'version' in k:
            del state[k]
            continue
        new_k = k.replace('self_attn_ln', 'self_attn_mid_layer_norm').\
                  replace('ffn_layernorm', 'ffn_layer_norm').\
                  replace('cross_attn_ln', 'cross_attn_mid_layer_norm').\
                  replace('encoder_attn', 'cross_attn').\
                  replace('attn_ln', 'self_attn_mid_layer_norm')
        v = state[k]
        del state[k]
        state[new_k] = v
    model["model"] = state
 

    ofa_config = OFAConfig(**config)
    model = OFAModel(ofa_config)
    remove_ignore_keys_(state,model.state_dict())
    model.load_state_dict(state)
    model.save_pretrained(hf_model)
    return model

            
def remove_ignore_keys_(state_dict,model_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)
    ignore_keys_prefix="encoder.embed_images"
    for k in list(state_dict.keys()):
        if ignore_keys_prefix in k:
            state_dict.pop(k, None)
    pretrained_dict = {k: v for k, v in model_dict.items() if k not in state_dict}
    state_dict.update(pretrained_dict)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert ofa original ckpt to huggingface.')
    parser.add_argument('--pt_model', type=str, default='',
                        help='path of original ckpt')
    parser.add_argument('--hf_model_dir', type=str, default='',
                        help='directory of huggingface ckpt')
    args = parser.parse_args()
    trans_fairseq_to_huggingface(args.pt_model, args.hf_model_dir, ofa_base)
