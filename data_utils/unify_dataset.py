# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
from torchvision import transforms
import base64
from io import BytesIO
from PIL import ImageFile
import csv
import h5py
from .vision_helper import RandomAugment
from .transforms import *
import math
import re
import logging
import contextlib
from .data_utils import read_img_general, read_text_general

from.ofa_dataset import OFADataset, get_whole_word_mask, continuous_tense, collate_fn

label_map = {'entailment': 0, 'not_entailment': 1}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class UnifyDataset(OFADataset):
    def __init__(self,
                 args,
                 dataset,
                 is_label=False,
                 is_test=False):
        super().__init__(args, dataset, is_test)
        self.is_label=is_label
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.pretrain_seed
        self.code_dict_size = args.code_dict_size
        self.num_bins = args.num_bins
        self.patch_image_size = args.patch_image_size
        self.code_image_size = args.code_image_size

        self.pure_text_dataset = args.pure_text_dataset
        self.pure_image_dataset = args.pure_image_dataset
        self.detection_dataset = args.detection_dataset
        self.epoch = 0

        # self.all_object_list = args.all_object_list
        # self.all_caption_list = args.all_caption_list
        # self.all_relation_list = args.all_relation_list
        # self.type2ans_dict = args.type2ans_dict
        # self.ans2type_dict = args.ans2type_dict

        # self.attr2type_dict = args.attr2type_dict
        # self.type2attr_dict = args.type2attr_dict

        # self.rel2cap = args.rel2cap
        # self.rel2question = args.rel2question

        self.remove_grounded_captioning = args.remove_grounded_captioning = False
        self.remove_visual_grounding = args.remove_visual_grounding = False

        self.mask_ratio = args.mask_ratio
        self.random_ratio = args.random_ratio
        self.keep_ratio = args.keep_ratio
        self.mask_length = args.mask_length
        self.poisson_lambda = args.poisson_lambda
        self.replace_length = args.replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_idx = args.tokenizer.mask_token_id

        self.src_dict = {value: key for key, value in args.tokenizer.get_vocab().items()}
        added = {value: key for key, value in args.tokenizer.get_added_vocab().items()}
        self.src_dict.update(added)

        self.mask_whole_word = (
            get_whole_word_mask(args.tokenizer, self.src_dict)
            if self.mask_length != "subword"
            else None
        )
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda
            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        # self.pos_tgt_item = self.encode_text(" yes")
        self.pos_tgt_item = self.tokenizer(" yes", return_tensors="pt",
                                      add_special_tokens=False).input_ids.squeeze(0)

        # self.neg_tgt_item = self.encode_text(" no")
        self.neg_tgt_item = self.tokenizer(" no", return_tensors="pt",
                                           add_special_tokens=False).input_ids.squeeze(0)

        self.mask_left = self.mask_top = int(0.5 * self.code_image_size)
        self.mask_right = self.mask_bottom = int(1.5 * self.code_image_size)
        self.mask_ids = [
            i*self.code_image_size*2+j
            for i in range(self.code_image_size*2) for j in range(self.code_image_size*2)
            if not (self.mask_left <= i < self.mask_right and self.mask_top <= j < self.mask_bottom)
        ]

        scales = np.arange(args.patch_image_size, 481).tolist()

        # for image-text pair
        self.detection_large_resolution_transform = Compose([
            RandomHorizontalFlip(),
            LargeScaleJitter(output_size=args.patch_image_size, aug_scale_min=1.0, aug_scale_max=1.5),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=args.patch_image_size)
        ])
        self.patch_resize_transform = transforms.Compose([
            FixResize(self.patch_image_size, max_size=self.patch_image_size),
            transforms.CenterCrop(self.patch_image_size),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for pure image
        self.patch_crop_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for detection
        self.detection_transform = Compose([
            RandomHorizontalFlip(),
            LargeScaleJitter(output_size=self.code_image_size*2, aug_scale_min=1.0, aug_scale_max=1.5),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=args.patch_image_size)
        ])
        # for visual grounding
        self.positioning_transform = self.visual_grounding_transform = Compose([
            FixResize(self.patch_image_size, max_size=self.patch_image_size),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=args.patch_image_size)
        ])
        # CenterCrop((args.patch_image_size, args.patch_image_size)),
        self.dataset = dataset

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])


        self.max_bbox_num = 4


    def pre_question(self, question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question

    def pre_caption(self, caption, max_words):
        caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def get_negative_object(self, object):
        negative_object = random.choice(self.all_object_list[:-1])
        negative_object = self.all_object_list[-1] if negative_object == object else negative_object
        return negative_object

    def get_negative_attribute(self, attr_value):
        neg_attr_type = self.attr2type_dict[attr_value]
        neg_attr_list = self.type2attr_dict[neg_attr_type]
        neg_attr_value = random.choice(neg_attr_list[:-1])
        neg_attr_value = neg_attr_list[-1] if neg_attr_value == attr_value else neg_attr_value
        return neg_attr_value

    def get_negative_relation(self, gt_relation_set):
        negative_relation_list = [
           negative_relation for negative_relation in self.all_relation_list if negative_relation not in gt_relation_set
        ]
        negative_relation = random.choice(negative_relation_list)
        return negative_relation

    def get_negative_caption(self, caption, overlap_objects=None):
        prob = random.random()
        if overlap_objects is not None and overlap_objects != '' and prob > 0.6:
            overlap_object = random.choice(overlap_objects.strip().split('&&'))
            negative_object = random.choice(self.all_object_list[:-1])
            negative_object = self.all_object_list[-1] if negative_object == overlap_object else negative_object
            negative_caption = caption.replace(overlap_object, negative_object)
        else:
            negative_caption = random.choice(self.all_caption_list)
        return negative_caption

    def get_negative_answer(self, answer, conf):
        prob = random.random()
        if conf > (prob + 0.1) and answer in self.ans2type_dict:
            negative_answer_type = self.ans2type_dict[answer]
            if negative_answer_type == 'how many' and answer.isdigit() and prob > 0.5:
                negative_answer = int(answer) + random.choice([-1, 1]) if answer != 0 else 1
            else:
                negative_answer_list = self.type2ans_dict[negative_answer_type]
                negative_answer = random.choice(negative_answer_list[:-1])
                negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
            return negative_answer

        negative_answer_list = self.type2ans_dict['other']
        negative_answer = random.choice(negative_answer_list[:-1])
        negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
        return negative_answer


    def process_image_text_pair(self, index):

        # uniq_id, caption, question, refs, gt_objects, predict_objects, \
        # overlap_objects, attribute, relation, image, dataset_name, type = self.dataset[index]

        # toler, ntoler = 0, 4
        # while toler < ntoler:
        #     try:
        #         uniq_id, img_id,img_path,caption, refs, gt_objects, dataset_name, type= self.dataset[index]
        
        #         # uniq_id, img_path, caption, question, refs, gt_objects, dataset_name, type = self.dataset[index]
        #         # image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        #         image = read_img_general(img_path).convert("RGB")
        #         break
            # except:
            #     index = random.randint(0, len(self.dataset) - 1)
            #     toler += 1
            #     with open('/mnt/lustre/suyulin/debug/ofa-hf-master/hf-ofa-base-semi/attr_bad_ceph.tsv', 'a') as f_bad:
            #         tsv_bad_logger = csv.writer(f_bad, delimiter='\t')
            #         tsv_bad_logger.writerow([uniq_id, img_id,img_path,caption, refs, gt_objects, dataset_name, type])
        uniq_id, img_id,img_path,caption, refs, gt_objects, dataset_name, type= self.dataset[index]
        image = read_img_general(img_path).convert("RGB")
        # patch_image = self.patch_resize_transform(image) if type not in ['visual_grounding', 'gvg', 'positioning','vg'] else None
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])
        pos_src_item = None
        neg_src_item = None

        if type == 'caption':
            tgt_caption = self.pre_caption(caption, self.max_tgt_length)
            pos_src_caption = self.pre_caption(caption, self.max_src_length)
            neg_src_caption = self.pre_caption(self.get_negative_caption(caption, gt_objects), self.max_src_length)

            src_item = self.tokenizer(" what does the image describe?", return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            tgt_item = self.tokenizer(" {}".format(tgt_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

            pos_src_item = self.tokenizer(' does the image describe " {} "?'.format(pos_src_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            neg_src_item = self.tokenizer(' does the image describe " {} "?'.format(neg_src_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)


        elif type == 'qa' or type == 'attr':
            question = self.pre_question(question, self.max_src_length)
            ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in refs.split('&&')}
            answer = max(ref_dict, key=ref_dict.get)
            conf = ref_dict[answer]
            src_item = self.tokenizer(" {}".format(question), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            tgt_item = self.tokenizer(" {}".format(answer), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

            conf = torch.tensor([conf])
            pos_src_item = self.tokenizer(
                ' what is the answer to question " {} ". is " {} "?'.format(question, answer), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            neg_src_item = self.tokenizer(
                ' what is the answer to question " {} ". is " {} "?'.format(question, self.get_negative_answer(answer, conf)), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

        elif type == 'attribute':
            object, attr_key, attr_value = attribute.strip().split('|!+')
            if attr_value == 'transparent':
                src_item = self.tokenizer(" what is transparent?", return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

                tgt_item = self.tokenizer(" {}".format(object), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

                pos_src_item = self.tokenizer(" is the {} transparent?".format(object), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

                neg_src_item = self.tokenizer(" is the {} transparent?".format(self.get_negative_object(object)), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)

            elif attr_value in {'plastic', 'textile', 'leather', 'wooden'}:
                rand_idx = random.randint(0, 1)
                if rand_idx == 0:
                    src_item = self.tokenizer(" what is the {} made of?".format(object), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                    tgt_item = self.tokenizer(" {}".format(attr_value), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                else:
                    src_item = self.tokenizer(" what is made of {}".format(attr_value), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                    tgt_item = self.tokenizer(" {}".format(object), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                pos_src_item = self.tokenizer(" is the {} made of {}?".format(object, attr_value), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                neg_src_item = self.tokenizer(
                    " is the {} made of {}?".format(object, self.get_negative_attribute(attr_value)), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)

            elif attr_value in {'stand', 'walk', 'run', 'jump', 'sit', 'lay'}:
                src_item = self.tokenizer(" what is the action of the {}?".format(object), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                tgt_item = self.tokenizer(" {}".format(continuous_tense(attr_value)), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                pos_src_item = self.tokenizer(
                    " is the {} {}?".format(object, continuous_tense(attr_value))
                    , return_tensors="pt",
                    add_special_tokens=False).input_ids.squeeze(0)
                neg_src_item = self.tokenizer(
                    " is the {} {}?".format(
                        object, continuous_tense(self.get_negative_attribute(attr_value))
                    )
                    , return_tensors="pt",
                    add_special_tokens=False).input_ids.squeeze(0)
            elif attr_value in {'smile', 'cry'}:
                src_item = self.tokenizer(" what expression is on the {}'s face?".format(object), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                tgt_item = self.tokenizer(" {}".format(continuous_tense(attr_value)), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                pos_src_item = self.tokenizer(
                    " is the {} {}?".format(object, continuous_tense(attr_value))
                    , return_tensors="pt",
                    add_special_tokens=False).input_ids.squeeze(0)
                neg_src_item = self.tokenizer(
                    " is the {} {}?".format(
                        object, continuous_tense(self.get_negative_attribute(attr_value))
                    )
                    , return_tensors="pt",
                    add_special_tokens=False).input_ids.squeeze(0)
            elif attr_value in {'sing', 'talk'}:
                src_item = self.tokenizer(" the {} is talking or singing?".format(object), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                tgt_item = self.tokenizer(" {}".format(continuous_tense(attr_value)), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
                pos_src_item = self.tokenizer(
                    " is the {} {}?".format(object, continuous_tense(attr_value))
                    , return_tensors="pt",
                    add_special_tokens=False).input_ids.squeeze(0)
                neg_src_item = self.tokenizer(
                    " is the {} {}?".format(
                        object, continuous_tense(self.get_negative_attribute(attr_value))
                    )
                    , return_tensors="pt",
                    add_special_tokens=False).input_ids.squeeze(0)
            else:
                raise NotImplementedError
        elif type == 'relation':
            object1, object2, relations = relation.strip().split('|!+')
            relation_list = relations.strip().split(', ')
            relation_set = set(relation_list)
            relation = random.choice(relation_list)
            src_item = self.tokenizer(' what is the relationship between " {} " and " {} "?'.format(object1, object2), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
            tgt_item = self.tokenizer(" {}".format(self.rel2cap[relation].format(object1, object2)), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
            pos_src_item = self.tokenizer(
                " {}".format(self.rel2question[relation].format(object1, object2))
                , return_tensors="pt",
                add_special_tokens=False).input_ids.squeeze(0)
            neg_src_item = self.tokenizer(
                " {}".format(self.rel2question[self.get_negative_relation(relation_set)].format(object1, object2))
                , return_tensors="pt",
                add_special_tokens=False).input_ids.squeeze(0)
        # elif type == 'positioning' or type == 'visual_grounding':
        #     conf = torch.tensor([0.0]) if self.remove_visual_grounding else torch.tensor([1.0])
        #     w, h = image.size
        #     boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        #     x0, y0, x1, y1 = refs.strip().split(',')
        #     boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        #     boxes_target["labels"] = np.array([0])
        #     boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])
        #     patch_image, boxes_target = self.positioning_transform(image, boxes_target)
        #     quant_x0 = "<bin_{}>".format(int((boxes_target["boxes"][0][0] * (self.num_bins - 1)).round()))
        #     quant_y0 = "<bin_{}>".format(int((boxes_target["boxes"][0][1] * (self.num_bins - 1)).round()))
        #     quant_x1 = "<bin_{}>".format(int((boxes_target["boxes"][0][2] * (self.num_bins - 1)).round()))
        #     quant_y1 = "<bin_{}>".format(int((boxes_target["boxes"][0][3] * (self.num_bins - 1)).round()))
        #     region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        #     src_caption = self.pre_caption(caption, self.max_src_length)
        #     src_item = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt",
        #                                       add_special_tokens=False).input_ids.squeeze(0)
        #     tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
        elif type == 'detection':
            w, h = image.size
            boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
            label_list = caption.strip().split('&&')
            for label_idx, label in enumerate(label_list):
                if label_idx >= self.max_bbox_num:
                    break
                x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
                boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
                boxes_target["labels"].append(cat)
                boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
            boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
            boxes_target["labels"] = np.array(boxes_target["labels"])
            boxes_target["area"] = torch.tensor(boxes_target["area"])
            patch_image, boxes_target = self.detection_large_resolution_transform(image, boxes_target)
            quant_boxes = []
            for i, box in enumerate(boxes_target["boxes"]):
                quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
                quant_boxes.extend(self.tokenizer.tokenize(' {}'.format(boxes_target["labels"][i])))
            src_item = self.tokenizer(' what are the objects in the image?', return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
            tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(quant_boxes))
            neg_src_caption = self.pre_caption(self.get_negative_caption('', None), self.max_src_length)
            neg_src_item = self.tokenizer(' does the image describe " {} "?'.format(neg_src_caption), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
        elif type == 'gvg' or type == 'visual_grounding'or type == 'vg':
            conf = torch.tensor([1.0])
            w, h = image.size
            boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
            label_list = gt_objects.strip().split(';')
            for i, label in enumerate(label_list):
                if i >= self.max_bbox_num:
                    break
                x0, y0, x1, y1 = label.strip().split(',')
                
                boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
                boxes_target["labels"].append(0)
                boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
            boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
            boxes_target["labels"] = np.array(boxes_target["labels"])
            boxes_target["area"] = torch.tensor(boxes_target["area"])

            patch_image, boxes_target_new = self.visual_grounding_transform(image, boxes_target)
            resize_h, resize_w = boxes_target_new["size"][0], boxes_target_new["size"][1]
            quant_boxes = []
            for i, box in enumerate(boxes_target_new["boxes"]):
                quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
            region_coord = ' '.join(quant_boxes)
            src_caption = self.pre_caption(refs, self.max_src_length)
            src_item = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt",
                                              add_special_tokens=False).input_ids.squeeze(0)
            # tgt_item = self.encode_text(region_coord, use_bpe=False)    
            if region_coord != '':
                tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
            else:
                tgt_item = None  
        else:
            logger = logging.getLogger("Main")
            logger.info('type {} not implement'.format(type))
            raise NotImplementedError

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item]) if tgt_item is not None else self.eos_item
        prev_output_item = torch.cat([self.bos_item, tgt_item]) if tgt_item is not None else self.bos_item
        pos_src_item = torch.cat([self.bos_item, pos_src_item, self.eos_item]) if pos_src_item is not None else None
        neg_src_item = torch.cat([self.bos_item, neg_src_item, self.eos_item]) if neg_src_item is not None else None

        if type == 'caption' and dataset_name in ['cc12m', 'CC12M']:
            target_item[:2] = self.tokenizer.pad_token_id
            target_item[-1] = self.eos_item

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            # "conf": conf,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            # "region_coord": boxes_target["boxes"]
        }

        examples = [example]
        prob = random.random()
        if type in ['positioning', 'visual_grounding', 'gvg','vg']:
            if region_coord != '' and type == 'vg':
                region_example = example.copy()
                region_prefix_item = self.tokenizer('  what does the region describe? region:', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                region_coord_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
                region_src_item = torch.cat([region_prefix_item, region_coord_item])
                region_tgt_item = self.tokenizer(' {}'.format(self.pre_caption(refs, self.max_tgt_length)), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                region_example["source"] = torch.cat([self.bos_item, region_src_item, self.eos_item])
                region_example["target"] = torch.cat([region_tgt_item, self.eos_item])
                region_example["prev_output_tokens"] = torch.cat([self.bos_item, region_tgt_item])
                # region_example["conf"] = torch.tensor([0.0]) if self.remove_grounded_captioning else torch.tensor([1.0])
                examples.append(region_example)
            else:
                pass
        elif prob >= 0.5 and not self.is_test and pos_src_item is not None:
            pos_example = example.copy()
            pos_example["source"] = pos_src_item
            pos_example["target"] = torch.cat([self.pos_tgt_item, self.eos_item])
            pos_example["prev_output_tokens"] = torch.cat([self.bos_item, self.pos_tgt_item])
            examples.append(pos_example)
        elif not self.is_test and neg_src_item is not None:
            neg_example = example.copy()
            neg_example["source"] = neg_src_item
            neg_example["target"] = torch.cat([self.neg_tgt_item, self.eos_item])
            neg_example["prev_output_tokens"] = torch.cat([self.bos_item, self.neg_tgt_item])
            examples.append(neg_example)
        return examples

    # def process_image_text_pair_unlabel_1(self, index):
    #     img_id,img_path = self.pure_image_dataset[index]
    #     image = read_img_general(img_path).convert("RGB")
    #     patch_mask = torch.tensor([True])
    #     w, h = image.size
    #     boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
    #     file_name_bbox='/mnt/lustre/share_data/suyulin/VG/refcocog/detections_total/boxes.hdf5'
    #     f= h5py.File(file_name_bbox,'r')
    #     label_list=f[img_id][:]
    #     for i, label in enumerate(label_list):
    #         if i >= self.max_bbox_num:
    #             break
    #         x0, y0, x1, y1 = label
    #         boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
    #         boxes_target["labels"].append(0)
    #         boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
    #     boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
    #     boxes_target["labels"] = np.array(boxes_target["labels"])
    #     boxes_target["area"] = torch.tensor(boxes_target["area"])
    #     patch_image,patch_boxes_target = self.visual_grounding_transform(image,boxes_target)
    #     resize_h, resize_w = patch_boxes_target["size"][0], patch_boxes_target["size"][1]
    #     quant_boxes = []
    #     for i, box in enumerate(patch_boxes_target["boxes"]):
    #         quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
    #     region_coord = ' '.join(quant_boxes)
    #     region_prefix_item = self.tokenizer('  what does the region describe? region:', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
    #     examples=[]
    #     for j in range(0, len(region_coord), 4):
    #         if j+4 > len(region_coord):
    #             break
    #         coord=region_coord[j: j+4]
    #         region_coord_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(coord.split()))
    #         region_src_item = torch.cat([region_prefix_item, region_coord_item])
    #         src_item = torch.cat([self.bos_item, region_src_item, self.eos_item])
    #         example = {
    #             # "id": img_id,
    #             "img_id": img_path,
    #             "source": src_item,
    #             "patch_image": patch_image,
    #             "patch_mask": patch_mask,
    #             "ref": region_coord_item,
    #             "prev_output_tokens" :  self.bos_item,
    #             "target" : self.eos_item,
    #             "w_resize_ratio": resize_w / w,
    #             "h_resize_ratio": resize_h / h
    #         }
    #         examples.append(example)
    #     return examples

    # def process_image_text_pair_unlabel_2(self, index):
    #     uniq_id, img_id,img_path,caption, refs, gt_objects, dataset_name, type= self.dataset[index]
        
    #     image = read_img_general(img_path).convert("RGB")
    #     patch_mask = torch.tensor([True])
    #     w, h = image.size
    #     patch_image = self.patch_resize_transform(image)
    #     resize_h, resize_w = patch_image["size"][0], patch_image["size"][1]
    #     src_caption = self.pre_caption(refs, self.max_src_length)       
    #     src_item = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt",
    #                                         add_special_tokens=False).input_ids.squeeze(0)
    #     src_item = torch.cat([self.bos_item, src_item, self.eos_item])
    #     ### vg task ### text -> bbox -> text
    #     example = {
    #         # "id": uniq_id,
    #         "img_id": img_path,
    #         "source": src_item,
    #         "patch_image": patch_image,
    #         "patch_mask": patch_mask,
    #         "ref": refs,
    #         "prev_output_tokens" :  self.bos_item,
    #         "target" : self.eos_item,
    #         "w_resize_ratio": resize_w / w,
    #         "h_resize_ratio": resize_h / h
    #     }
    #     return example

    def process_image_text_pair_unlabel(self, index):
        uniq_id, img_id,img_path,caption, refs, gt_objects, dataset_name, type= self.dataset[index]
        
        image = read_img_general(img_path).convert("RGB")
        patch_mask = torch.tensor([True])
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        file_name_bbox='/mnt/lustre/share_data/suyulin/VG/refcocog/detections_total/boxes.hdf5'
        
        f= h5py.File(file_name_bbox,'r')
        label_list=f[img_id][:]
        random_item=random.randint(1,min(len(label_list),self.max_bbox_num))
        x0, y0, x1, y1=label_list[random_item-1]
        # for i, label in enumerate(label_list):
        #     if i == random_item:
        # x0, y0, x1, y1 = label
        boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["labels"].append(0)
        boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
        boxes_target["labels"] = np.array(boxes_target["labels"])
        boxes_target["area"] = torch.tensor(boxes_target["area"])
        patch_image,patch_boxes_target = self.visual_grounding_transform(image,boxes_target)
        resize_h, resize_w = patch_boxes_target["size"][0], patch_boxes_target["size"][1]
        quant_boxes = []
        for i, box in enumerate(patch_boxes_target["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
        region_coord = ' '.join(quant_boxes)
        src_caption = self.pre_caption(refs, self.max_src_length)       
        src_item = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt",
                                            add_special_tokens=False).input_ids.squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        tgt_item = self.tokenizer(' {}'.format(self.pre_caption(refs, self.max_tgt_length)), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
    
        ### vg task ### text -> bbox -> text
        example = {
            "id": uniq_id,
            "img_id": img_path,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "ref": refs,
            "prev_output_tokens" :  self.bos_item,
            "target" : self.eos_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h
        }
        examples = [example]
        ### gc task ### bbox -> text -> bbox
        if region_coord != '':
            region_example = example.copy()
            region_prefix_item = self.tokenizer('  what does the region describe? region:', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            region_coord_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
            region_src_item = torch.cat([region_prefix_item, region_coord_item])
            region_example["source"] = torch.cat([self.bos_item, region_src_item, self.eos_item])
            # new_label_list=[]
            # for label in boxes_target["boxes"]:
            #     label=','.join(str(int(l.item() * 100) / 100) for l in label)
            #     new_label_list.append(label)
            # region_ref_item= ';'.join(l for l in new_label_list)#torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
            # # print(region_ref_item)
            ref_label=','.join(str(int(l.item() * 100) / 100) for l in label_list[random_item-1])
            region_example["ref"]= ref_label
            examples.append(region_example)
        return examples


    def process_pure_text(self, index):
        patch_image = torch.zeros((3, self.code_image_size*2, self.code_image_size*2))
        patch_mask = torch.tensor([False])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        examples = []
        for _ in range(2):
            uniq_id, text = self.pure_text_dataset[index]
            text = text.strip().lower()
            text_item = self.tokenizer(" {}".format(text), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)[:512]

            text_item = text_item[-256:]
            text_item = torch.cat([self.bos_item, text_item, self.eos_item])
            mask_text_item = self.add_whole_word_mask(text_item.clone(), self.mask_ratio)
            prefix_item = self.tokenizer(' what is the complete text of " "?', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            src_item = torch.cat([prefix_item[:-2], mask_text_item[1:-1], prefix_item[-2:]])
            tgt_item = text_item[1:-1]
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            target_item = torch.cat([tgt_item, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "code_mask": code_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "conf": conf,
            }
            examples.append(example)

        return examples

    def process_pure_image(self, index):
        image_id, image, text, code, dataset_name = self.pure_image_dataset[index]
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        image = read_img_general(image).convert("RGB")
        patch_image = self.patch_crop_transform(image)
        patch_image[:, 64:192, 64:192] = 0
        patch_mask = torch.tensor([True])
        if dataset_name in ('imagenet_22k', 'yfcc100m', 'oi'):
            src_item = self.tokenizer(" what is the image in the middle part?", return_tensors="pt",
                           add_special_tokens=False).input_ids.squeeze(
                0)
        else:
            caption = self.pre_caption(text, self.max_src_length)
            src_item = self.tokenizer(" what is the image in the middle part? caption: {}".format(caption), return_tensors="pt",
                           add_special_tokens=False).input_ids.squeeze(
                0)
        image_code = torch.LongTensor([int(num) for num in code.strip().split()])
        tgt_item = image_code + len(self.src_dict) - self.code_dict_size - self.num_bins
        code_mask = torch.tensor([True])
        conf = torch.tensor([2.0])

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return [example]

    def process_detection(self, index):
        image_id, image, label = self.detection_dataset[index]
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        image = read_img_general(image).convert("RGB")

        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        label_list = label.strip().split('&&')
        for label_idx, label in enumerate(label_list):
            if label_idx >= self.max_bbox_num:
                break
            x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
            boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
            boxes_target["labels"].append(cat)
            boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
        boxes_target["labels"] = np.array(boxes_target["labels"])
        boxes_target["area"] = torch.tensor(boxes_target["area"])

        patch_image, boxes_target = self.detection_transform(image, boxes_target)
        patch_mask = torch.tensor([True])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        quant_boxes = []
        for i, box in enumerate(boxes_target["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
            quant_boxes.extend(self.tokenizer.tokenize(' {}'.format(boxes_target["labels"][i])))
        src_item = self.tokenizer(' what are the objects in the image?', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(
            0)
        tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(quant_boxes))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return [example]

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch):
            if self.is_label:
                pair_samples = self.process_image_text_pair(index)
            else:
                pair_samples= self.process_image_text_pair_unlabel(index) 

        return pair_samples


    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go oevr the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                # 4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                4, self.tokenizer.vocab_size, size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        # 4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                        4, self.tokenizer.vocab_size, size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        # 4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                        4, self.tokenizer.vocab_size, size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=4, high=self.tokenizer.vocab_size, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    # def collate(self, samples):
    #     """Merge samples of different tasks to form two mini-batches.
    #     Args:
    #         samples (List[Tuple]): samples to collate
    #     Returns:
    #         Tuple[dict]: two mini-batch containing the data of different tasks
    #     """

    #     samples_v1 = []   # containing image-text pairs
    #     samples_v2 = []   # containing detection data, text data and image data
    #     for sample_tuple in samples:
    #         samples_v1 += sample_tuple[0]
    #         samples_v2 += sample_tuple[1]

    #     if samples_v2 != []:
    #         res_v1 = collate_fn(samples_v1, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
    #         res_v2 = collate_fn(samples_v2, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
    #         return res_v1, res_v2
    #     else:
    #         res_v1 = collate_fn(samples_v1, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
    #         return res_v1






        # if samples_v2 == []:
        #     samples_v2 += self.process_pure_text(0) if self.pure_text_dataset else []
        #     samples_v2 += self.process_pure_image(0) if self.pure_image_dataset else []
        #     samples_v2 += self.process_detection(0) if self.detection_dataset else []

        # res_v1 = collate_fn(samples_v1, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
        # res_v2 = collate_fn(samples_v2, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
        # return res_v1, res_v2
        # # return res_v1
