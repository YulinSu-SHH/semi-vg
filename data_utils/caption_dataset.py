# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import string
from torchvision import transforms
import base64
import re
from io import BytesIO
from PIL import Image, ImageFile
from .transforms import *
import contextlib
from .ofa_dataset import OFADataset
from .data_utils import read_img_general, read_text_general

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





class CaptionDataset(OFADataset):
    def __init__(self,
                 args,
                 dataset,
                 is_label=True,
                 is_test=False):
        super().__init__(args, dataset, is_test)

        if args.imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.scst = args.scst
        self.transtab = str.maketrans({key: None for key in string.punctuation})
        self.max_tgt_length = args.max_tgt_length
        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.dataset = dataset
        self.is_label=is_label
        self.patch_image_size = args.patch_image_size
        self.max_image_size = args.max_image_size
        self.num_bins = args.num_bins
        self.max_bbox_num = 4

        self.positioning_transform = Compose([
            FixResize(self.patch_image_size, max_size=self.patch_image_size),
            ToTensor(),
            Normalize(mean=mean, std=std, max_image_size=self.patch_image_size)
        ])

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

    def __getitem__(self, index):
        uniq_id, img_id,img_path,caption, refs, gt_objects, dataset_name, type= self.dataset[index]
        image = read_img_general(img_path).convert("RGB")
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])
        pos_src_item = None
        neg_src_item = None

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

        patch_image, boxes_target_new = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = boxes_target_new["size"][0], boxes_target_new["size"][1]
        quant_boxes = []
        for i, box in enumerate(boxes_target_new["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
        region_coord = ' '.join(quant_boxes)
        region_prefix_item = self.tokenizer('  what does the region describe? region:', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        region_coord_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
        region_src_item = torch.cat([region_prefix_item, region_coord_item])
        region_tgt_item = self.tokenizer(' {}'.format(self.pre_caption(refs, self.max_tgt_length)), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        src_item = torch.cat([self.bos_item, region_src_item, self.eos_item])
        target_item = torch.cat([region_tgt_item, self.eos_item])
        prev_output_item= torch.cat([self.bos_item, region_tgt_item])

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
        return example



def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:
        def is_beginning_of_word(i):
            # if i < dictionary.nspecial:
            if i < 4:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            # print(i, tok)
            if tok.startswith("madeupword"):
                return True
            try:
                # print(i, tok, bpe.convert_tokens_to_string(tok), bpe.convert_tokens_to_string(tok).startswith(" "))
                return bpe.convert_tokens_to_string(tok).startswith(" ")
            except ValueError:
                # print("wrong")
                return True






# # Copyright 2022 The OFA-Sys Team.
# # All rights reserved.
# # This source code is licensed under the Apache 2.0 license
# # found in the LICENSE file in the root directory.

# from inspect import ArgSpec
# import string
# from torchvision import transforms
# import base64
# from io import BytesIO
# from PIL import Image, ImageFile
# from .transforms import *
# import contextlib
# from .ofa_dataset import OFADataset
# from .data_utils import read_img_general, read_text_general
# import re
# import os
# label_map = {'entailment': 0, 'not_entailment': 1}

# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# ImageFile.LOAD_TRUNCATED_IMAGES = True
# ImageFile.MAX_IMAGE_PIXELS = None
# Image.MAX_IMAGE_PIXELS = None


# @contextlib.contextmanager
# def numpy_seed(seed, *addl_seeds):
#     """Context manager which seeds the NumPy PRNG with the specified seed and
#     restores the state afterward"""
#     if seed is None:
#         yield
#         return
#     if len(addl_seeds) > 0:
#         seed = int(hash((seed, *addl_seeds)) % 1e6)
#     state = np.random.get_state()
#     np.random.seed(seed)
#     try:
#         yield
#     finally:
#         np.random.set_state(state)





# class CaptionDataset(OFADataset):
#     def __init__(self,
#                  args,
#                  dataset,
#                  is_label=True,
#                  is_test=False):
#         super().__init__(args, dataset, is_test)
#         self.patch_image_size = args.patch_image_size
#         self.max_image_size = args.max_image_size
#         self.max_bbox_num = 4
#         if args.imagenet_default_mean_and_std:
#             mean = IMAGENET_DEFAULT_MEAN
#             std = IMAGENET_DEFAULT_STD
#         else:
#             mean = [0.5, 0.5, 0.5]
#             std = [0.5, 0.5, 0.5]

#         self.positioning_transform = Compose([
#             FixResize(self.patch_image_size, max_size=self.patch_image_size),
#             ToTensor(),
#             Normalize(mean=mean, std=std, max_image_size=self.patch_image_size)
#         ])

#         # self.patch_resize_transform = transforms.Compose([
#         #     lambda image: image.convert("RGB"),
#         #     transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=Image.BICUBIC),
#         #     transforms.ToTensor(),
#         #     transforms.Normalize(mean=mean, std=std),
#         # ])

#         self.scst = args.scst
#         self.transtab = str.maketrans({key: None for key in string.punctuation})
#         self.max_tgt_length = args.max_tgt_length
#         self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
#         self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
#         self.dataset = dataset
#         self.is_label=is_label
#         self.tokenizer=args.tokenizer
#         self.max_src_length = args.max_src_length
#         self.num_bins = args.num_bins

    
#     def pre_caption(self, caption, max_words):
#         caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

#         caption = re.sub(
#             r"\s{2,}",
#             ' ',
#             caption,
#         )
#         caption = caption.rstrip('\n')
#         caption = caption.strip(' ')

#         # truncate caption
#         caption_words = caption.split(' ')
#         if len(caption_words) > max_words:
#             caption = ' '.join(caption_words[:max_words])

#         return caption

#     def __getitem__(self, index):
#         uniq_id, img_id,img_path,caption, text, gt_objects, dataset_name, type = self.dataset[index]
#         # uniq_id, image, caption = self.dataset[index]
#         image = read_img_general(img_path).convert("RGB")
#         patch_mask = torch.tensor([True])
#         conf = torch.tensor([1.0])
#         pos_src_item = None
#         neg_src_item = None
#         w, h = image.size
#         boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
#         label_list = gt_objects.strip().split(';')
#         for i, label in enumerate(label_list):
#             if i >= self.max_bbox_num:
#                 break
#             x0, y0, x1, y1 = label.strip().split(',')
#             boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
#             boxes_target["labels"].append(0)
#             boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
#         boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
#         boxes_target["labels"] = np.array(boxes_target["labels"])
#         boxes_target["area"] = torch.tensor(boxes_target["area"])
#         patch_image, boxes_target_new = self.positioning_transform(image, boxes_target)
#         resize_h, resize_w = boxes_target_new["size"][0], boxes_target_new["size"][1]
#         # patch_image = self.patch_resize_transform(image)
#         quant_boxes = []
#         for i, box in enumerate(boxes_target_new["boxes"]):
#             quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
#         region_coord = ' '.join(quant_boxes)

#         prefix_item = self.tokenizer('  what does the region describe? region:', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
#         coord_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
#         src_item = torch.cat([prefix_item, coord_item])
#         tgt_item = self.tokenizer(' {}'.format(self.pre_caption(text, self.max_tgt_length)), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        
#         # if not self.is_test and not self.scst:
#         #     caption = caption.translate(self.transtab).strip()
#         #     caption_token_list = caption.strip().split()
#         #     tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
#         # else:
#         #     caption = ' '.join(caption.strip().split())
#         #     caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
#         #     tgt_caption = '&&'.join(caption_list)
#         # src_item = self.tokenizer(" what does the image describe?", return_tensors="pt",
#         #                           add_special_tokens=False).input_ids.squeeze(0)
#         # tgt_item = self.tokenizer(" {}".format(tgt_caption), return_tensors="pt",
#         #                           add_special_tokens=False).input_ids.squeeze(0)

#         src_item = torch.cat([self.bos_item, src_item, self.eos_item])
#         target_item = torch.cat([tgt_item, self.eos_item])
#         prev_output_item = torch.cat([self.bos_item, tgt_item])

#         example = {
#             "id": uniq_id,
#             "img_id": img_path,
#             "source": src_item,
#             "patch_image": patch_image,
#             "patch_mask": patch_mask,
#             "target": target_item,
#             "prev_output_tokens": prev_output_item,
#             "w_resize_ratio": resize_w / w,
#             "h_resize_ratio": resize_h / h,
#             "region_coord": boxes_target["boxes"]
#         }
#         return example


# def get_whole_word_mask(bpe, dictionary):
#     if bpe is not None:
#         def is_beginning_of_word(i):
#             # if i < dictionary.nspecial:
#             if i < 4:
#                 # special elements are always considered beginnings
#                 return True
#             tok = dictionary[i]
#             # print(i, tok)
#             if tok.startswith("madeupword"):
#                 return True
#             try:
#                 # print(i, tok, bpe.convert_tokens_to_string(tok), bpe.convert_tokens_to_string(tok).startswith(" "))
#                 return bpe.convert_tokens_to_string(tok).startswith(" ")
#             except ValueError:
#                 # print("wrong")
#                 return True



