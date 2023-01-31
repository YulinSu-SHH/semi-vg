from PIL import Image
from torchvision import transforms
import sys
sys.path.append("/mnt/cache/zhangzhao2/codes/ofa-hf")

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

set_seed(2022)

pre_mean, pre_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 384
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=pre_mean, std=pre_std)
])

print(f'resolution is {resolution}')

def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    
    bin_list += ["<bin_{}>".format(int(round(coord_list[0] * w_resize_ratio /512 * (1000 - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[1] * h_resize_ratio /512 * (1000 - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[2] * w_resize_ratio /512 * (1000 - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[3] * h_resize_ratio /512 * (1000 - 1))))]
  
    return ' '.join(bin_list)

def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [bin_list[0] / (1000 - 1) * 512 / w_resize_ratio]
    coord_list += [bin_list[1] / (1000 - 1) * 512 / h_resize_ratio]
    coord_list += [bin_list[2] / (1000 - 1) * 512 / w_resize_ratio]
    coord_list += [bin_list[3] / (1000 - 1) * 512 / h_resize_ratio]
    return coord_list


if __name__ == '__main__':

    input_parser = argparse.ArgumentParser(description='OFA zero')
    input_parser.add_argument('--sampling', default=False , type=None, help="whether sampling decoder")
    input_parser.add_argument('--beam', default=5, type=int, help="beam size")
    input_parser.add_argument('--no_repeat_size', default=0 , type=int, help="beam size")
    input_parser.add_argument('--ref', default='motorbike with a lot of people' , type=str, help="referring")
    input_parser.add_argument('--predictory', default='/mnt/lustre/lujinghui1/events/data/' , type=str, help="referring")
    input_parser.add_argument('--val_dir', default='/mnt/lustre/lujinghui1/events/anno/carry3people/val.jsonl' , type=str, help="validation gt data dir")
    input_parser.add_argument('--save_dir', default='./zeroshot_carry3people' , type=str, help="output dir")
    input_parser.add_argument('--model', default='/mnt/lustre/lujinghui1/ofa_models/OFA_base' , type=str, help="model")
    input_parser.add_argument('--tokenizer_path', default='/mnt/cache/zhangzhao2/codes/ofa-hf/tokenizer' , type=str, help="model")
    input_parser.add_argument('--gpu', default=0 , type=int, help="gpu index")

    args = input_parser.parse_args()
    print(args)

    device = f'cuda:{args.gpu}'

    ## load model and tokenizer
    tokenizer = OFATokenizer.from_pretrained(args.tokenizer_path)
    model = OFAModel.from_pretrained(args.model, use_cache=False).to(device)

    # config model for outputing scores
    model.config.output_scores = True
    model.config.return_dict_in_generate = True 

    ## referring tokenizer and encode
    ref = args.ref
    txt = f" which region does the text ' {ref} ' describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids.to(device)

    items = []
    with open(args.val_dir,'r') as fin:
        for line_idx, line in enumerate(fin):
            items.append(json.loads(line))

    ## inference results
    predictions = []

    num_multi = num_neg = 0
    for item in tqdm(items[:]): 
        img_file = item['filename']

        if ref == "null ref":
            ref = item['caption']['sent']

        old_img_file = img_file[:]
        if "mnt" in img_file or "s3://" in img_file:
            pass
        else:
            img_file = f'{args.predictory}{img_file}'

        img = read_img_general(img_file)
        # img = Image.open(img_file)

        w, h = img.size
        w_resize_ratio = resolution/ w
        h_resize_ratio = resolution / h

        patch_img = patch_resize_transform(img).unsqueeze(0).to(device)
        gen = model.generate(inputs, patch_images=patch_img, num_beams=args.beam, no_repeat_ngram_size=args.no_repeat_size,num_return_sequences=1) 
        outputs = tokenizer.batch_decode(gen['sequences'], skip_special_tokens=True)
        
        # convert location token to coordinates
        res = outputs[0]
        res = res.split(' ')
        out_bbox_list = [' '.join(res[i: i+4]) for i in range(0, len(res), 4) if i+4 <= len(res)]
        if len(out_bbox_list) > 1:
            num_multi += 1
        if len(out_bbox_list) == 0:
            num_neg += 1
        for out_bbox in out_bbox_list:
            coord_list = bin2coord(f'{out_bbox}', w_resize_ratio, h_resize_ratio)
            if args.beam == 1:
                token_scores = []
                for iraw_prob in gen['scores']:
                    iraw_prob = F.softmax(iraw_prob, dim=1)
                    token_scores.append(torch.max(iraw_prob).item())
                score = np.mean(token_scores)
            else:
                score = gen['sequences_scores'].item()
            predictions.append({"image_id":old_img_file,"bbox":coord_list,"label":1,"score":score,"ref":ref})
        # except Exception:
            # print(outputs)
            # coord_list = bin2coord(f'{outputs[0][:4]}', w_resize_ratio, h_resize_ratio)
            # predictions.append({"image_id":old_img_file,"bbox":coord_list,"label":1,"score":gen['sequences_scores'].item(),"ref":ref})

    print(f'num_multi: {num_multi}\n', f'num_neg: {num_neg}')

    name_ref = '_'.join(args.ref.split(' '))
    model_ref = args.model.split('/')[-1]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    

    if args.sampling:
        with open(f'{args.save_dir}/predictions_{args.beam}_[{name_ref}]_{model_ref}.jsonl', 'w') as f:
            for line in predictions:
                json.dump(line, f,ensure_ascii=False)
                f.write('\n')
    else:
        with open(f'{args.save_dir}/beam_predictions_{args.beam}_[{name_ref}]_{model_ref}.jsonl', 'w') as f:
            for line in predictions:
                json.dump(line, f,ensure_ascii=False)
                f.write('\n')

   