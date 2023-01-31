from asyncio import tasks
from gc import get_objects
import imp
import os
import random
import time
import string
import json
import numpy as np
import torch
import csv
from sklearn.metrics import roc_auc_score
from torch.nn.parallel.distributed import DistributedDataParallel
from textbrewer.distiller_utils import move_to_device
from evaluation import bin2coord
from data_utils.input_dataset import FileDataset as inputDataset
from data_utils.unify_dataset import UnifyDataset
from utils import MyConcatDataset
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Pool, pool
from data_utils.input_dataset import FileDataset
def generate_pseudo_loader(world_size,loader_unlabel,model,args,device,logger,out_tsv_paths):
    for out_tsv in out_tsv_paths:
        generate_pseudo_tsv(world_size,loader_unlabel,model,args,device,logger,out_tsv)
    pseudo_dataset = [inputDataset(out_tsv_paths[i], args.selected_cols) for i in range(len(out_tsv_paths))]
    dataset_pseudo = [UnifyDataset(args,pseudo_dataset[i],is_label=True) for i in range(len(out_tsv_paths))]
    dataset_pseudo = MyConcatDataset(args,dataset_pseudo)
    sampler_pseudo = DistributedSampler(dataset_pseudo)
    loader_unlabel = torch.utils.data.DataLoader(dataset_pseudo,
                                                sampler=sampler_pseudo,
                                                collate_fn=dataset_pseudo.collate,
                                                batch_size=args.batch_size,
                                                drop_last=True,
                                                num_workers=0,
                                                pin_memory=True)
    return loader_unlabel
def generate_pseudo_tsv(world_size,loader_unlabel,model,args,device,logger,out_tsv):
    
    task=os.path.split(out_tsv)[-1].split('_')[0]
    f_out= open(out_tsv, 'a',newline="")     
    tsv_writer = csv.writer(f_out,delimiter='\t')
    forward_time = 0
    generate_time = 0
    cnt=0
    if task=='caption':
        img_id_list=[]
        for idx, data in enumerate(loader_unlabel):
            for i in data:
                data_gc=data[1]
            data_gc = move_to_device(data_gc, device)
            batch_gc = data_gc["net_input"]
            with torch.no_grad():
                forward_time_start = time.time()
                forward_time_end = time.time()
                forward_time += forward_time_end - forward_time_start
                forward_gen_start = time.time()
                if args.generator_version == "hf":
                    gen_output = model.generate(input_ids=batch_gc["input_ids"],patch_images=batch_gc["patch_images"],
                                                    patch_masks=batch_gc["patch_masks"],
                                                    num_beams=args.beam,
                                                    max_length=args.max_len_b,
                                                    min_length=args.min_len,
                                                    no_repeat_ngram_size=args.no_repeat_ngram_size)
                elif args.generator_version == "fairseq":
                    with torch.no_grad():
                        try:
                            gen_output = args.generator.generate([model], data_gc)
                        except:
                            gen_output = args.generator.generate([model.module], data_gc)
                        gen_output = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

                forward_gen_end = time.time()
                generate_time += forward_gen_end-forward_gen_start
                decode_tokens = args.tokenizer.batch_decode(gen_output,skip_special_tokens=True)
                transtab = str.maketrans({key: None for key in string.punctuation})
                for i in range(len(gen_output)):
                    hyp=decode_tokens[i].translate(transtab).strip()
                    uniq_id=data_gc['id'][i].item()
                    image_path=data_gc['img_id'][i]
                    image_id=image_path.split('/')[-1][:-4]
                    caption=' '
                    refs=hyp
                    gt_objects=data_gc['refs'][i]
                    dataset_name='unlabel'
                    type='vg'
                    if image_id not in img_id_list:
                        res_line=[uniq_id, image_id, image_path,caption, refs,gt_objects, dataset_name, type]
                        for i in range(world_size):
                            if torch.distributed.get_rank()==i:
                                tsv_writer.writerow(res_line)
                        cnt+=8
                        img_id_list.append(image_id)
                    if cnt%1000==0:
                        logger.info(f"Generate {cnt} samples")

    elif task == 'vg':
        for idx, data in enumerate(loader_unlabel):
            data_vg=data[0]
            data_vg = move_to_device(data_vg, device)
            batch_vg = data_vg["net_input"]
            with torch.no_grad():
                forward_time_start = time.time()
                forward_time_end = time.time()
                forward_time += forward_time_end - forward_time_start
                forward_gen_start = time.time()
                if args.generator_version == "hf":
                    constraint_start, constraint_end = args.constraint_range.split(',')
                    constraint_start = int(constraint_start)
                    constraint_end = int(constraint_end)
                    bad_words_ids = [[x] for x in list(range(4,constraint_start))+list(range(constraint_end,len(args.src_dict)))]
                    gen_output = model.generate(input_ids=batch_vg["input_ids"], patch_images=batch_vg["patch_images"],
                                                patch_masks=batch_vg["patch_masks"],
                                                num_beams=args.beam,
                                                max_length=args.max_len_b+2,
                                                min_length=args.min_len+2,
                                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                bad_words_ids=bad_words_ids)
                    gen_output = [x[1:] for x in gen_output]
                elif args.generator_version == "fairseq":
                    with torch.no_grad():
                        try:
                            gen_output_ori = args.generator.generate([model], data_vg)
                        except:
                            gen_output_ori = args.generator.generate([model.module], data_vg)
                        gen_output = [gen_output_ori[i][0]['tokens'] for i in range(len(gen_output_ori))]                   

                forward_gen_end = time.time()
                generate_time += forward_gen_end-forward_gen_start

                for i in range(len(gen_output)):
                    res = args.tokenizer.batch_decode(gen_output[i][:-1], skip_special_tokens=True) 
                    out_bbox_list= [' '.join(res[j: j+4]) for j in range(0, len(res), 4) if j+4 <= len(res)]           
                    # coord_list_new=[]
                    for out_bbox in out_bbox_list:
                        try:
                            coord_list = bin2coord(f'{out_bbox}', data_vg['w_resize_ratios'][i], data_vg['h_resize_ratios'][i])
                            coord_item=','.join(str(int(c.item() * 100) / 100) for c in coord_list)
                        except:
                            print(out_bbox)
                            break
                    
                    # out_coords=';'.join(c for c in coord_list_new)
                    if coord_item!='':
                        uniq_id=data_vg['id'][i]
                        image_path=data_vg['img_id'][i]
                        image_id=image_path.split('/')[-1][:-4]
                        caption=' '
                        refs=data_vg['refs'][i]
                        gt_objects=coord_item
                        dataset_name='unlabel'
                        type='vg'
                        res_line=[uniq_id, image_id, image_path,caption, refs,gt_objects, dataset_name, type]
                        for i in range(world_size):
                            if torch.distributed.get_rank()==i:
                                tsv_writer.writerow(res_line)
                        cnt+=8
                    if cnt%1000==0:
                        logger.info(f"Generate {cnt} samples")
    
    f_out.close()
    logger.info(
            f"Total generate time : {generate_time}; Second per sample : {generate_time/(cnt+0.0000001)}; Total sample: {cnt}"
        )