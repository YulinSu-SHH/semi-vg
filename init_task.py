from generate import sequence_generator
from metrics.ciderD import CiderD
from ofa.tokenization_ofa import OFATokenizer
import torch
from data_utils.snli_ve_dataset import Trie
from data_utils.input_dataset import FileDataset as inputDataset
import pickle
import json, os
TOKENIZER_PATH = "./tokenizer"

def build_task(args):
    tokenizer = OFATokenizer.from_pretrained(TOKENIZER_PATH)
    args.tokenizer = tokenizer
    args.tgt_dict = args.src_dict = {value: key for key, value in tokenizer.get_vocab().items()}
    paths = args.tables.split(",")


    args.test_dataset = inputDataset(paths[-1], args.selected_cols, data_slice=False)
    args.best_score = 0
    if args.task == 'pretrain':
        args.pure_text_dataset = inputDataset(args.text_data, args.text_selected_cols) if args.text_data else None
        args.pure_image_dataset = inputDataset(args.image_data, args.image_selected_cols, capacity=512) if args.image_data else None
        args.detection_dataset = inputDataset(args.detection_data, args.detection_selected_cols) if args.detection_data else None
        args.train_dataset = [inputDataset(paths[i], args.selected_cols) for i in range(len(paths)-1)]
        args.generator = sequence_generator.SequenceGenerator(tokenizer=tokenizer,
                                                                  beam_size=args.beam,
                                                                  max_len_b=args.max_len_b,
                                                                  min_len=args.min_len,
                                                                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                                  constraint_range=args.constraint_range)

        args.CiderD_scorer = CiderD(df=args.eval_cider_cached_tokens)
        # args.all_object_list = [
        #     row.strip() for row in open(os.path.join(args.neg_sample_dir, 'object.txt')) if row.strip() != ''
        # ]

        # args.all_caption_list = [
        #     row.strip() for row in open(os.path.join(args.neg_sample_dir, 'all_captions.txt')) if row.strip() != ''
        # ]

        # args.type2attr_dict = {
        #     'material': ['plastic', 'textile', 'leather', 'wooden'],
        #     'action': ['stand', 'walk', 'run', 'jump', 'sit', 'lay'],
        #     'expression': ['smile', 'cry'],
        #     'other': ['sing', 'talk']
        # }
        # args.attr2type_dict = {value: key for key, values in args.type2attr_dict.items() for value in values}

        # args.type2ans_dict = json.load(open(os.path.join(args.neg_sample_dir, 'type2ans.json')))
        # args.ans2type_dict = {}
        # for type, answer_list in args.type2ans_dict.items():
        #     if type == 'other':
        #         continue
        #     for answer in answer_list:
        #         args.ans2type_dict[answer] = type
                
        if os.path.exists(os.path.join(args.neg_sample_dir, 'relation.txt')):
            args.all_relation_list = [
                row.strip() for row in open(os.path.join(args.neg_sample_dir, 'relation.txt')) if row.strip() != ''
            ]
            args.rel2cap = {}
            args.rel2question = {}
            for relation in args.all_relation_list:
                if relation in {'at', 'on', 'inside of', 'interacts with', 'under'}:
                    args.rel2cap[relation] = '{} ' + relation + ' {}'
                    args.rel2question[relation] = 'is {} ' + relation + ' {}?'
                elif relation in {'holds', 'wears', 'ride', 'dance', 'plays'}:
                    args.rel2cap[relation] = '{} ' + relation[:-1] + 'ing' + ' {}'
                    args.rel2question[relation] = 'is {} ' + relation[:-1] + 'ing' + ' {}?'
                elif relation in {'eat', 'cut', 'hug'}:
                    args.rel2cap[relation] = '{} ' + relation + relation[-1] + 'ing' + ' {}'
                    args.rel2question[relation] = 'is {} ' + relation + relation[-1] + 'ing' + ' {}?'
                elif relation in {'surf', 'hang', 'drink', 'skateboard', 'catch', 'kiss', 'throw', 'snowboard', 'kick',
                                'ski',
                                'read'}:
                    args.rel2cap[relation] = '{} ' + relation + 'ing' + ' {}'
                    args.rel2question[relation] = 'is {} ' + relation + 'ing' + ' {}?'
                elif relation == 'holding hands':
                    args.rel2cap[relation] = '{} ' + 'holding hands with' + ' {}'
                    args.rel2question[relation] = 'is {} ' + 'holding hands with' + ' {}?'
                elif relation == 'contain':
                    args.rel2cap[relation] = '{} ' + 'contains' + ' {}'
                    args.rel2question[relation] = 'is {} ' + 'contains' + ' {}?'
                elif relation == 'talk on phone':
                    args.rel2cap[relation] = '{} ' + 'and' + ' {}' + ' are talking on phone'
                    args.rel2question[relation] = 'are {} ' + 'and' + ' {}' + ' talking on phone?'
                elif relation == 'hits':
                    args.rel2cap[relation] = '{} ' + 'hitting' + ' {}'
                    args.rel2question[relation] = 'is {} ' + 'hitting' + ' {}?'
                elif relation == 'highfive':
                    args.rel2cap[relation] = '{} ' + 'and' + ' {}' + ' are high fives'
                    args.rel2question[relation] = 'are {} ' + 'and' + ' {}' + ' high fives?'
                elif relation == 'handshake':
                    args.rel2cap[relation] = '{} ' + 'and' + ' {}' + ' are shaking hands'
                    args.rel2question[relation] = 'are {} ' + 'and' + ' {}' + ' shaking hands?'
                else:
                    raise NotImplementedError
        else:
            args.all_relation_list = []
            args.rel2cap = {}
            args.rel2question = {}
    elif args.task in ['caption_stage1', 'caption_stage2']:
        args.pure_text_dataset = inputDataset(args.text_data, args.text_selected_cols) if args.text_data else None
        args.pure_image_dataset = inputDataset(args.image_data, args.image_selected_cols, capacity=512) if args.image_data else None
        args.detection_dataset = inputDataset(args.detection_data, args.detection_selected_cols) if args.detection_data else None
        args.train_dataset = [inputDataset(paths[i], args.selected_cols) for i in range(len(paths) - 1)]
        if args.generator_version == 'fairseq':
            args.generator = sequence_generator.SequenceGenerator(tokenizer=tokenizer,
                                                                  beam_size=args.beam,
                                                                  max_len_b=args.max_len_b,
                                                                  min_len=args.min_len,
                                                                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                                  constraint_range=args.constraint_range)

        args.CiderD_scorer = CiderD(df=args.eval_cider_cached_tokens)
    elif args.task in ['refcoco', 'refcocog', 'refcocoplus']:
        args.pure_text_dataset = inputDataset(args.text_data, args.text_selected_cols) if args.text_data else None
        args.pure_image_dataset = inputDataset(args.image_data, args.image_selected_cols, capacity=512) if args.image_data else None
        args.detection_dataset = inputDataset(args.detection_data, args.detection_selected_cols) if args.detection_data else None
        args.train_dataset = [inputDataset(paths[i], args.selected_cols) for i in range(len(paths) - 1)]
        if args.generator_version == 'fairseq':
            args.generator = sequence_generator.SequenceGenerator(tokenizer=tokenizer,
                                                                  beam_size=args.beam,
                                                                  max_len_b=args.max_len_b,
                                                                  min_len=args.min_len,
                                                                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                                  constraint_range=args.constraint_range)
    elif args.task in ['snli_ve']:
        args.train_dataset = [inputDataset(paths[i], args.selected_cols) for i in range(len(paths) - 1)]
        answer_item_list = []
        args.index2ans = {}
        args.ans2label_dict = {"no": 0, "yes": 1, "maybe": 2}
        args.constraint_trie = Trie(tokenizer.eos_token_id)
        for i, answer in enumerate(args.ans2label_dict.keys()):
            answer_item = tokenizer(' ' + answer, return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze(0)
            answer_item_list.append(answer_item)
            args.index2ans[i] = answer
            args.constraint_trie.insert([tokenizer.bos_token_id] + answer_item.tolist() + [tokenizer.eos_token_id])

        constraint_mask_list = []
        for answer_item in answer_item_list:
            constraint_mask = torch.zeros((len(answer_item) + 1, len(args.tgt_dict))).bool()
            for i in range(len(answer_item) + 1):
                constraint_prefix_token = [tokenizer.bos_token_id] + answer_item[:i].tolist()
                constraint_nodes = args.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)
        args.valid_answers_list = []
        args.valid_constraint_masks_list = []
        for i in range(0, len(answer_item_list), args.batch_size):
            args.valid_answers_list += [answer_item_list[i:i + args.batch_size]]
            args.valid_constraint_masks_list += [constraint_mask_list[i:i + args.batch_size]]
    elif args.task in ['vqa_gen']:
        args.train_dataset = [inputDataset(paths[i], args.selected_cols) for i in range(len(paths) - 1)]
        if args.ans2label_file is not None:
            args.ans2label_dict = pickle.load(open(args.ans2label_file, "rb"))
        else:
            args.ans2label_dict = {"no": 0, "yes": 1}
        print("ans2label_dict", args.ans2label_dict)
        answer_item_list = []
        args.index2ans = {}
        args.constraint_trie = Trie(tokenizer.eos_token_id)
        for i, answer in enumerate(args.ans2label_dict.keys()):
            answer_item = tokenizer(' ' + answer, return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze(0)
            answer_item_list.append(answer_item)
            args.index2ans[i] = answer
            args.constraint_trie.insert([tokenizer.bos_token_id] + answer_item.tolist() + [tokenizer.eos_token_id])

        constraint_mask_list = []
        for answer_item in answer_item_list:
            constraint_mask = torch.zeros((len(answer_item) + 1, len(args.tgt_dict))).bool()
            for i in range(len(answer_item) + 1):
                constraint_prefix_token = [tokenizer.bos_token_id] + answer_item[:i].tolist()
                constraint_nodes = args.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        if args.val_inference_type == "allcand":
            args.valid_answers_list = []
            args.valid_constraint_masks_list = []
            for i in range(0, len(answer_item_list), args.batch_size):
                args.valid_answers_list += [answer_item_list[i:i + args.batch_size]]
                args.valid_constraint_masks_list += [constraint_mask_list[i:i + args.batch_size]]
        elif args.val_inference_type == "beamsearch":
            args.generator = sequence_generator.SequenceGenerator(tokenizer=tokenizer,
                                                                  beam_size=args.beam,
                                                                  max_len_b=args.max_len_b,
                                                                  min_len=args.min_len,
                                                                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                                  constraint_range=args.constraint_range,
                                                                  normalize_scores=False)
    else:
        raise NotImplementedError
