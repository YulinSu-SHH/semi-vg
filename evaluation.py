import math
from traceback import print_tb
import torch
from collections import defaultdict
from utils import save_checkpoint
from criterions import AdjustLabelSmoothedCrossEntropyCriterion
import string
import time
import os
from data_utils.ofa_dataset import collate_tokens
from typing import Dict, List, Optional
from textbrewer.distiller_utils import move_to_device
import csv
from metrics.evaluator import Evaluator
import numpy as np
from metrics.pycocoevalcap.eval import COCOEvalCap
from metrics.cider import calculate_cider_scores
import torch.nn.functional as F
import pipeline
TOKENIZER_PATH = "./tokenizer"


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
    coord_list += [bin_list[0] / (1000 - 1) * 256 / w_resize_ratio]
    coord_list += [bin_list[1] / (1000 - 1) * 256 / h_resize_ratio]
    coord_list += [bin_list[2] / (1000 - 1) * 256 / w_resize_ratio]
    coord_list += [bin_list[3] / (1000 - 1) * 256 / h_resize_ratio]
    return coord_list


def get_perplexity(outputs=None, **kwargs):
    assert 'loss' in outputs
    perplexity = math.exp(torch.mean(outputs["loss"]))
    return {"perplexity": perplexity}


def ddp_evaluate(model, step, eval_dataloader, args, logger):
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info(f"Do predict in local rank : {args.local_rank}")
        evaluate(model, step, eval_dataloader, args, logger)
        args.evaluate_idx += 1
        if args.local_rank != -1:  # DDP is enabled
            torch.distributed.barrier()
    else:
        torch.distributed.barrier()


def evaluate(model, step, eval_dataloader, args, logger):
    try:
        model = model.module
    except:
        model = model
    tokenizer = args.tokenizer
    device = args.device

    if args.generator_version == "fairseq":
        try:
            generator = args.generator
        except:
            logger.info("Don't need generator when evaluation.")

    if args.rank == 0:
        logger.info("Evaluation...")
    model.eval()
    outputs = defaultdict(list)
    eval_res_list=[]
    adaptor = pipeline.get_adaptors(args)
    if args.task in ['caption_stage1', 'caption_stage2','pretrain']:
        eval_dataloader_gc=eval_dataloader[1]
        cider_sum = 0.0
        cider_cnt = 0
        forward_time = 0
        generate_time = 0
        cnt=0
        # hyps, refs = dict(), dict()
        # caption_evaluator=COCOEvalCap()
        hyps, refs = [],[]
        confidence_list=[]
        score_list=[]

        for idx, data in enumerate(eval_dataloader_gc):
            
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                forward_time_start = time.time()
                forward_time_end = time.time()
                forward_time += forward_time_end - forward_time_start
                forward_gen_start = time.time()
                #######
                
                # results = model(input_ids=batch["input_ids"],patch_images=batch["patch_images"],
                #                                    patch_masks=batch["patch_masks"],decoder_input_ids=batch["decoder_input_ids"])
                # results = adaptor(data, results)
                
                # for loss in results["batch_losses"]:
                #     confidence_list.append(loss.data.item())
                
                ########
                if args.generator_version == "hf":
                    gen_output = model.generate(input_ids=batch["input_ids"],patch_images=batch["patch_images"],
                                                   patch_masks=batch["patch_masks"],
                                                   num_beams=args.beam,
                                                   max_length=args.max_len_b,
                                                   min_length=args.min_len,
                                                   no_repeat_ngram_size=args.no_repeat_ngram_size)
                elif args.generator_version == "fairseq":
                    with torch.no_grad():
                        gen_output_ori = generator.generate([model], data)
                        gen_output = [gen_output_ori[i][0]["tokens"] for i in range(len(gen_output_ori))]

                forward_gen_end = time.time()
                generate_time += forward_gen_end-forward_gen_start
                
                decode_tokens = tokenizer.batch_decode(gen_output,skip_special_tokens=True)
                target_decode_tokens = tokenizer.batch_decode(data["target"],skip_special_tokens=True)
                # hyps, refs = [],[]
                transtab = str.maketrans({key: None for key in string.punctuation})
                for i in range(len(gen_output)):
                    # hyps[data['id'][i]]=decode_tokens[i].translate(transtab).strip()
                    # refs[data['id'][i]]=target_decode_tokens[i].translate(transtab).strip()
                    gen_hyp=decode_tokens[i].translate(transtab).strip()
                    hyps.append(gen_hyp)
                    gen_ref=[sent.translate(transtab).strip() for sent in target_decode_tokens[i].split('&&')]
                    refs.append(gen_ref)
                    ########
                    
                    scores_single = calculate_cider_scores([gen_hyp], [gen_ref],args.CiderD_scorer)
                    score_list.append(scores_single)
                    
                    #######
                    cnt+=1
                
        scores = calculate_cider_scores(hyps, refs,args.CiderD_scorer)
        cider_sum += scores.sum()
        cider_cnt += int(scores.size)
   
                # if idx%100==0:
                #     logger.info("example hypothesis: " + hyps[i])
                #     logger.info("example reference: " + ' && '.join(refs[i]))
       
        for idx in range(cnt):
            logger.info("example hypothesis: " + hyps[idx])
            logger.info("example reference: " + ' && '.join(refs[idx]))
            logger.info(f"example conficence: {score_list[idx].sum()}" )
            # logger.info(f"example conficence: {confidence_list[idx]} \ {score_list[idx].sum()}" )
        
        
        #     # caption_evaluator.evaluate(refs,hyps)
        eval_res = {"cider": cider_sum/cider_cnt}
        eval_res_list.append(eval_res)
        current_score = eval_res[args.metric_gc]
        logger.info(
            f"Total generate time : {generate_time}; Second per sample : {generate_time/(cider_cnt+0.0000001)}; Total sample: {cider_cnt}"
        )
        if args.metric_gc == "cider":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    if args.task in ['refcoco','refcocog','refcocoplus','vg','pretrain']:
        eval_dataloader_vg=eval_dataloader[0]
        score_sum = 0.0
        score_cnt = 0
        forward_time = 0
        generate_time = 0
        
        predictedBoxList=[]
        gtBoxList=[]
        predictedBoxList_xy=[]
        gtBoxList_xy=[]
        confidence_list=[]
        score_list=[]
        cnt=0
        for idx, data in enumerate(eval_dataloader_vg):
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                forward_time_start = time.time()
                forward_time_end = time.time()
                forward_time += forward_time_end - forward_time_start
                forward_gen_start = time.time()
                ###########
                
                # results = model(input_ids=batch["input_ids"],patch_images=batch["patch_images"],
                #                                    patch_masks=batch["patch_masks"],decoder_input_ids=batch["decoder_input_ids"])
                # results = adaptor(data, results)
                # for loss in results["batch_losses"]:
                #     confidence_list.append(loss.data.item())
                
                ###########
                if args.generator_version == "hf":
                    constraint_start, constraint_end = args.constraint_range.split(',')
                    constraint_start = int(constraint_start)
                    constraint_end = int(constraint_end)
                    bad_words_ids = [[x] for x in list(range(4,constraint_start))+list(range(constraint_end,len(args.src_dict)))]
                    gen_output = model.generate(input_ids=batch["input_ids"], patch_images=batch["patch_images"],
                                                patch_masks=batch["patch_masks"],
                                                num_beams=args.beam,
                                                max_length=args.max_len_b+2,
                                                min_length=args.min_len+2,
                                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                bad_words_ids=bad_words_ids)
                    gen_output = [x[1:] for x in gen_output]
                elif args.generator_version == "fairseq":
                    with torch.no_grad():
                        gen_output_ori = generator.generate([model], data)
                        gen_output = [gen_output_ori[i][0]['tokens'] for i in range(len(gen_output_ori))]              
  
                forward_gen_end = time.time()
                generate_time += forward_gen_end-forward_gen_start

                # hyps, refs = [], []
                
                for i in range(len(gen_output)):
                    res = tokenizer.batch_decode(gen_output[i][:-1], skip_special_tokens=True) 
                    out_bbox_list= [' '.join(res[j: j+4]) for j in range(0, len(res), 4) if j+4 <= len(res)]          
                    ref =tokenizer.batch_decode(data["target"][i][:-1], skip_special_tokens=True) #sample:  ['<bin_0>', '<bin_0>', '<bin_749>', '<bin_749>']
                    ref=' '.join(i for i in ref)
                    for out_bbox in out_bbox_list:
                        try:
                            coord_list = bin2coord(f'{out_bbox}', data['w_resize_ratios'][i], data['h_resize_ratios'][i])
                        except:
                            print(out_bbox)
                            break
                        x0,y0,x1,y1=coord_list
                        w=x1-x0
                        h=y1-y0
                        coord_wh=torch.tensor([x0,y0,w,h]).to(device)
                        coord_xy=torch.tensor([x0,y0,x1,y1]).to(device)
                        ref_coord_list = bin2coord(f'{ref}', data['w_resize_ratios'][i], data['h_resize_ratios'][i])
                        ref_x0,ref_y0,ref_x1,ref_y1=ref_coord_list
                        ref_w=ref_x1-ref_x0
                        ref_h=ref_y1-ref_y0
                        ref_coord_wh=torch.tensor([ref_x0,ref_y0,ref_w,ref_h]).to(device)
                        ref_coord_xy=torch.tensor([ref_x0,ref_y0,ref_x1,ref_y1]).to(device)
                        predictedBoxList.append(coord_wh)
                        gtBoxList.append(ref_coord_wh)
                        predictedBoxList_xy.append(coord_xy)
                        gtBoxList_xy.append(ref_coord_xy)
                        ######
                        evaluator_single = Evaluator()
                        (accuracy_single, iouList_single) = evaluator_single.evaluate([coord_wh], [ref_coord_wh], iouThreshold=0.5)
                        
                        confidence_list.append(iouList_single[0])
                        score_list.append(accuracy_single)
                        ######
                        
                        cnt+=1
                # if idx % 100 == 0:
                #     logger.info(f"example hypothesis: {predictedBoxList_xy[cnt-1]}"  )
                #     logger.info(f"example reference: {gtBoxList_xy[cnt-1]}" )


                #     hyps.append(gen_output[i][:-1] - len(args.src_dict) + args.num_bins)
                #     refs.append(data["target"][i][:-1] - len(args.src_dict) + args.num_bins)


                # hyps_tensor, refs_tensor = torch.stack(hyps, dim=0), torch.stack(refs, dim=0)
                # refs_tensor=refs_tensor.squeeze(1)
                
                # print(hyps_tensor.size(),refs_tensor.size())
                # hyps_tensor = hyps_tensor / (args.num_bins - 1) * args.max_image_size
                # refs_tensor = refs_tensor / (args.num_bins - 1) * args.max_image_size
                # hyps_tensor[:, ::2] /= data['w_resize_ratios'].unsqueeze(1)
                # hyps_tensor[:, 1::2] /= data['h_resize_ratios'].unsqueeze(1)
                # refs_tensor[:, ::2] /= data['w_resize_ratios'].unsqueeze(1)
                # refs_tensor[:, 1::2] /= data['h_resize_ratios'].unsqueeze(1)


                def calculate_ap_score(hyps, refs, thresh=0.5):
                    interacts = torch.cat(
                        [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
                         torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
                        dim=1
                    )
                    area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
                    area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
                    interacts_w = interacts[:, 2] - interacts[:, 0]
                    interacts_h = interacts[:, 3] - interacts[:, 1]
                    area_interacts = interacts_w * interacts_h
                    ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
                    return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

                # scores = calculate_ap_score(hyps_tensor,refs_tensor)
                # score_sum += scores.sum().item()
                # score_cnt += scores.size(0)
        # if idx % 100 == 0:
        for idx in range(cnt):
            logger.info(f"example hypothesis: {predictedBoxList_xy[idx]}"  )
            logger.info(f"example reference: {gtBoxList_xy[idx]}" )
            
            logger.info(f"example conficence: {confidence_list[idx]} \ {score_list[idx]}" )
        
            
            evaluator = Evaluator()
        (accuracy, iouList) = evaluator.evaluate(predictedBoxList, gtBoxList, iouThreshold=0.5)
        logger.info(f"Acc@0.5 : {accuracy}")
        eval_res = {"Acc@0.5": accuracy}
        eval_res_list.append(eval_res)
        current_score = eval_res[args.metric_vg]
        # logger.info(
        #     f"Total generate time : {generate_time}; Second per sample : {generate_time/(score_cnt)}"
        # )
        # logger.info(
        #     f"Total score : {score_sum}; Total sample: {score_cnt}"
        # )
        if args.metric_vg == "ap":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    if args.task in ['snli_ve']:
        score_sum = 0.0
        score_cnt = 0
        for idx, data in enumerate(eval_dataloader):
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                encoder_out = model.encoder(
                    batch["input_ids"],
                    patch_images=batch["patch_images"],
                    patch_masks=batch["patch_masks"]
                )
                valid_result = []
                eos_item = torch.LongTensor([tokenizer.eos_token_id])
                for valid_answers, valid_constraint_masks in zip(args.valid_answers_list, args.valid_constraint_masks_list):
                    valid_size = len(valid_answers)
                    valid_tgt_items = [
                        torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
                        for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                    ]
                    valid_prev_items = [
                        torch.cat([torch.tensor(decoder_prompt), valid_answer])
                        for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                    ]
                    valid_constraint_mask_items = [
                        torch.cat(
                            [torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(),
                             valid_constraint_mask],
                            dim=0
                        )
                        for decoder_prompt in data["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
                    ]
                    valid_tgt = collate_tokens(valid_tgt_items, pad_idx=tokenizer.pad_token_id).to(device)
                    valid_prev_output = collate_tokens(valid_prev_items, pad_idx=tokenizer.pad_token_id).to(device)
                    valid_constraint_masks = collate_tokens(valid_constraint_mask_items, pad_idx=tokenizer.pad_token_id).to(device)

                    new_encoder_out = {}
                    new_encoder_out["last_hidden_state"] = encoder_out["last_hidden_state"].repeat_interleave(valid_size, dim=0)
                    new_encoder_out["padding_mask"] = encoder_out["padding_mask"].repeat_interleave(valid_size, dim=0)
                    new_encoder_out["position_embedding"] = encoder_out["position_embedding"].repeat_interleave(valid_size, dim=0)


                    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
                        r"""
                        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
                        """
                        bsz, src_len = mask.size()
                        tgt_len = tgt_len if tgt_len is not None else src_len

                        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
                        return expanded_mask.masked_fill(expanded_mask.bool(), torch.finfo(dtype).min)

                    encoder_attention_mask = _expand_mask(new_encoder_out["padding_mask"], new_encoder_out["last_hidden_state"].dtype,
                                                          valid_prev_output.shape[-1])

                    decoder_out = model.decoder(valid_prev_output, encoder_hidden_states=new_encoder_out["last_hidden_state"],
                                                encoder_attention_mask=encoder_attention_mask,
                                                src_pos_embed=new_encoder_out["position_embedding"]
                                                )

                    decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                    lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
                    scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                    scores = scores.masked_fill(valid_tgt.eq(tokenizer.pad_token_id), 0)
                    scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
                    scores = scores.sum(1)
                    scores = scores.view(-1, valid_size)
                    valid_result.append(scores)
            valid_result = torch.cat(valid_result, dim=-1)
            predicts = valid_result.argmax(1).tolist()
            hyps = [args.index2ans[predict_index] for predict_index in predicts]
            results = [{"uniq_id": id, "answer": hyp} for id, hyp in zip(data["id"].tolist(), hyps)]
            scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(data['ref_dict'], hyps)]
            score_sum += sum(scores)
            score_cnt += len(scores)
            if idx % 10 == 0:
                logger.info(f"example hypothesis: {hyps[0]}")
                logger.info(f"example reference: {data['ref_dict'][0]}")
        score = score_sum / score_cnt
        score = score if isinstance(score, float) else score.item()
        score = round(score, 4)
        eval_res = {"acc": score}
        current_score = eval_res[args.metric]
        logger.info(
            f"Total sample: {score_cnt}"
        )
        if args.metric == "acc":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    if args.task in ['vqa_gen']:
        score_sum = 0.0
        score_cnt = 0
        for idx, data in enumerate(eval_dataloader):
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                if args.val_inference_type == "beamsearch":
                    hypos = args.generator.generate([model], data, prefix_tokens=data['prefix_tokens'])
                    hypos = [hypos[i][0]["tokens"] for i in range(len(hypos))]
                    results = []
                    for i, sample_id in enumerate(data["id"].tolist()):
                        prefix_len = data['prefix_tokens'][i].ne(1).sum().item()
                        detok_hypo_str = tokenizer.batch_decode(hypos[i][0]["tokens"][prefix_len:], skip_special_tokens=True)
                        results.append({"question_id": int(sample_id), "answer": detok_hypo_str.strip()})
                    scores = [ref_dict.get(result['answer'], 0) for ref_dict, result in
                              zip(data['ref_dict'], results)]
                elif args.val_inference_type == "allcand":
                    encoder_out = model.encoder(
                        batch["input_ids"],
                        patch_images=batch["patch_images"],
                        patch_masks=batch["patch_masks"]
                    )
                    eos_item = torch.tensor([tokenizer.eos_token_id])
                    pad = tokenizer.pad_token_id
                    valid_result = []
                    for valid_answers, valid_constraint_masks in zip(args.valid_answers_list,
                                                                     args.valid_constraint_masks_list):
                        valid_size = len(valid_answers)
                        valid_tgt_items = [
                            torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
                            for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                        ]
                        valid_prev_items = [
                            torch.cat([torch.tensor(decoder_prompt), valid_answer])
                            for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                        ]
                        valid_constraint_mask_items = [
                            torch.cat([torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(),
                                       valid_constraint_mask], dim=0)
                            for decoder_prompt in data["decoder_prompts"] for valid_constraint_mask in
                            valid_constraint_masks
                        ]
                        valid_tgt = collate_tokens(valid_tgt_items, pad_idx=pad, left_pad=False).to(device)
                        valid_prev_output = collate_tokens(valid_prev_items, pad_idx=pad, left_pad=False).to(device)
                        valid_constraint_masks = collate_tokens(valid_constraint_mask_items, pad_idx=pad,
                                                                left_pad=False).to(device)
                        new_encoder_out = {}
                        new_encoder_out["last_hidden_state"] = encoder_out["last_hidden_state"].repeat_interleave(valid_size, dim=0)

                        new_encoder_out["padding_mask"] = encoder_out["padding_mask"].repeat_interleave(valid_size, dim=0)

                        new_encoder_out["position_embedding"] = encoder_out["position_embedding"].repeat_interleave(valid_size, dim=0)


                        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
                            r"""
                            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
                            """
                            bsz, src_len = mask.size()
                            tgt_len = tgt_len if tgt_len is not None else src_len

                            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
                            return expanded_mask.masked_fill(expanded_mask.bool(), torch.finfo(dtype).min)

                        encoder_attention_mask = _expand_mask(new_encoder_out["padding_mask"],
                                                              new_encoder_out["last_hidden_state"].dtype,
                                                              valid_prev_output.shape[-1])

                        decoder_out = model.decoder(valid_prev_output, encoder_hidden_states= new_encoder_out["last_hidden_state"],
                                                    encoder_attention_mask=encoder_attention_mask,
                                                    src_pos_embed=new_encoder_out["position_embedding"]
                                                    )
                        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
                        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                        scores = scores.masked_fill(valid_tgt.eq(tokenizer.pad_token_id), 0)
                        scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
                        scores = scores.sum(1)
                        scores = scores.view(-1, valid_size)
                        valid_result.append(scores)
                    valid_result = torch.cat(valid_result, dim=-1)
                    predicts = valid_result.argmax(1).tolist()
                    hyps = [args.index2ans[predict_index] for predict_index in predicts]
                    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(data['ref_dict'], hyps)]
                score_sum += sum(scores)
                score_cnt += len(scores)
                if idx % 100 == 0:
                    logger.info(f"example hypothesis: {hyps[0]}")
                    logger.info(f"example reference: {data['ref_dict'][0]}")
        score = score_sum / score_cnt
        score = score if isinstance(score, float) else score.item()
        score = round(score, 4)
        eval_res = {"acc": score}
        current_score = eval_res[args.metric]
        logger.info(
            f"Total sample: {score_cnt}"
        )
        if args.metric == "acc":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    if args.task in ['pretrain_']:
        losses = []
        sample_size = 0
        for idx, data in enumerate(eval_dataloader):
            data[0] = move_to_device(data[0], device)
            data[1] = move_to_device(data[1], device)
            batch_v1 = data[0]["net_input"]
            batch_v2 = data[1]["net_input"]
            with torch.no_grad():
                outputs = [model(**batch_v1), model(**batch_v2)]
                criterion = AdjustLabelSmoothedCrossEntropyCriterion(args)
                loss, sample_size_i, logging_output = criterion(outputs, data)
                losses.append(loss.unsqueeze(0))
                sample_size += sample_size_i
        loss = torch.cat(losses, dim=0)
        loss = torch.sum(loss)
        eval_res = {"loss": loss / sample_size}
        current_score = eval_res[args.metric]
        if current_score < args.best_score:
            args.best_score = current_score
            args.best_step = step
            save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            idx = args.evaluate_idx % args.keep_last_ckpt_num
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step_%d" % idx))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))
    # else:
    #     raise NotImplementedError(
    #         f"The eval metric of task {args.task} is not implemented.")

    for eval_res in eval_res_list:
        res_string = "\t".join([f"{k}: {v:.6f}" for k, v in eval_res.items()])

        if args.rank == 0:
            logger.info(f"Eval results at {step} steps:")
            logger.info(f"{res_string}")
            # logger.info(
            #     f"Current best {args.metric}: {args.best_score} at {args.best_step} step"
            # )
    model.train()
