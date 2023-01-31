import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import pdb
import gradio as gr

from ofa.modeling_ofa import OFAModel
from ofa.tokenization_ofa import OFATokenizer
from transformers import set_seed
from tqdm import tqdm
import json, argparse, os
import numpy as np
from data_utils import read_img_general
import pdb
import torch
import torch.nn.functional as F
from generate import sequence_generator

set_seed(2022)

tokenizer_path = '/mnt/cache/zhangzhao2/codes/ofa-hf/tokenizer'
model_path = '/mnt/lustre/share_data/zhangzhao2/VG/log/hf-ofa-base-attr/pretrain/1015_2352/gs59366'

pre_mean, pre_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 384
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=pre_mean, std=pre_std)
])


## load model and tokenizer
tokenizer = OFATokenizer.from_pretrained(tokenizer_path)
model = OFAModel.from_pretrained(model_path, use_cache=False)

# config model for outputing scores
model.config.output_scores = True
model.config.return_dict_in_generate = True 


## generate
generator = sequence_generator.SequenceGenerator(
    tokenizer=tokenizer,
    beam_size=5,
    max_len_b=128,
    min_len=0,
    no_repeat_ngram_size=3,
)


# Move models to GPU
# use_cuda = torch.cuda.is_available()
# if use_cuda:
#     device = f'cuda:0'
#     model = model.to(device)


def interface(image, instruction):
    instruction = ' ' + instruction.strip() + ' '
    # Construct input sample & preprocess for GPU if cuda available
    inputs = tokenizer([instruction], return_tensors="pt").input_ids   

    # Generate result
    with torch.no_grad():
        patch_img = patch_resize_transform(image).unsqueeze(0)
        data = {}
        data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}

        gen_output = generator.generate([model], data)
        gen = gen_output[0][0]
        res = tokenizer.batch_decode(gen['tokens'], skip_special_tokens=True)

    return ' '.join(res[:-1])


