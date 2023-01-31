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
from generate import sequence_generator

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

if __name__ == '__main__':

    input_parser = argparse.ArgumentParser(description='OFA zero')
    input_parser.add_argument('--sampling', default=False , type=None, help="whether sampling decoder")
    input_parser.add_argument('--beam', default=5, type=int, help="beam size")
    input_parser.add_argument('--no_repeat_size', default=0 , type=int)
    input_parser.add_argument('--ref', default='motorbike with a lot of people' , type=str, help="referring")
    input_parser.add_argument('--predictory', default='/mnt/lustre/lujinghui1/events/data/' , type=str, help="referring")
    input_parser.add_argument('--val_dir', default='/mnt/lustre/lujinghui1/events/anno/carry3people/val.jsonl' , type=str, help="validation gt data dir")
    input_parser.add_argument('--save_dir', default='./zeroshot_carry3people' , type=str, help="output dir")
    input_parser.add_argument('--model', default='/mnt/lustre/lujinghui1/ofa_models/OFA_base' , type=str, help="model")
    input_parser.add_argument('--tokenizer_path', default='/mnt/cache/zhangzhao2/codes/ofa-hf/tokenizer' , type=str, help="model")
    input_parser.add_argument('--min_len', default=0 , type=int, help="min_len of output seq")
    input_parser.add_argument('--max_len_b', default=128 , type=int, help="min_len of output seq")
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


    ## generate
    generator = sequence_generator.SequenceGenerator(
        tokenizer=tokenizer,
        beam_size=args.beam,
        max_len_b=args.max_len_b,
        min_len=args.min_len,
        no_repeat_ngram_size=args.no_repeat_size,
    )

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
        # gen = model.generate(inputs, patch_images=patch_img, num_beams=args.beam, no_repeat_ngram_size=args.no_repeat_size,num_return_sequences=1) 
        data = {}
        data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}

        gen_output = generator.generate([model], data)
        gen = gen_output[0][0]
        res = tokenizer.batch_decode(gen['tokens'], skip_special_tokens=True)


        
        out_bbox_list = [' '.join(res[i: i+4]) for i in range(0, len(res), 4) if i+4 <= len(res)]
        if len(out_bbox_list) > 1:
            num_multi += 1
        if len(out_bbox_list) == 0:
            num_neg += 1
        for out_bbox in out_bbox_list:
            coord_list = bin2coord(f'{out_bbox}', w_resize_ratio, h_resize_ratio)
            score = gen['score'].item()
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

   