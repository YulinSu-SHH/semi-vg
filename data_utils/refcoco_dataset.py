# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


import base64
from io import BytesIO
from PIL import Image, ImageFile
from .transforms import *
import re
import logging
import warnings
from .ofa_dataset import OFADataset
from .data_utils import read_img_general, read_text_general


ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RefcocoDataset(OFADataset):
    def __init__(self,
                 args,
                 dataset,
                 is_label=False,
                 is_test=False):
        super().__init__(args, dataset, is_test)
        self.is_label=is_label
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length
        self.patch_image_size = args.patch_image_size
        self.max_image_size = args.max_image_size
        self.num_bins = args.num_bins
        self.max_bbox_num = 4

        if args.imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.positioning_transform = Compose([
            FixResize(self.patch_image_size, max_size=self.patch_image_size),
            ToTensor(),
            Normalize(mean=mean, std=std, max_image_size=self.patch_image_size)
        ])
        logging.info(f"imagenet_default_mean_and_std {args.imagenet_default_mean_and_std}, mean {mean}, std {std}")
        logging.info(f"patch_image_size {self.patch_image_size}, max_image_size {self.max_image_size}")
        self.dataset = dataset

        self.src_dict = {value: key for key, value in args.tokenizer.get_vocab().items()}
        added = {value: key for key, value in args.tokenizer.get_added_vocab().items()}
        self.src_dict.update(added)

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])

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
        # print("new: ",boxes_target_new['boxes'])
        resize_h, resize_w = boxes_target_new["size"][0], boxes_target_new["size"][1]
        quant_boxes = []
        for i, box in enumerate(boxes_target_new["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
        region_coord = ' '.join(quant_boxes)
        src_caption = self.pre_caption(refs, self.max_src_length)
        src_item = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt",
                                            add_special_tokens=False).input_ids.squeeze(0)
        
        if region_coord != '':
            tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
        else:
            tgt_item = None  
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
            "img_id": img_path,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            # "region_coord": region_coord#boxes_target["boxes"]
        }
        return example

        # example = {
        #     "id": uniq_id,
        #     "source": src_item,
        #     "patch_image": patch_image,
        #     "patch_mask": patch_mask,
        #     "target": target_item,
        #     "prev_output_tokens": prev_output_item,
        #     "conf": conf,
        # }

        # examples = [example]
        # prob = random.random()
        # if type in ['positioning', 'visual_grounding', 'gvg','vg']:
        #     if region_coord != '' and type == 'vg':
        #         region_example = example.copy()
        #         region_prefix_item = self.tokenizer('  what does the region describe? region:', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        #         region_coord_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))
        #         region_src_item = torch.cat([region_prefix_item, region_coord_item])
        #         region_tgt_item = self.tokenizer(' {}'.format(self.pre_caption(caption, self.max_tgt_length)), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        #         region_example["source"] = torch.cat([self.bos_item, region_src_item, self.eos_item])
        #         region_example["target"] = torch.cat([region_tgt_item, self.eos_item])
        #         region_example["prev_output_tokens"] = torch.cat([self.bos_item, region_tgt_item])
        #         region_example["conf"] = torch.tensor([0.0]) if self.remove_grounded_captioning else torch.tensor([1.0])
        #         examples.append(region_example)
        #     else:
        #         pass
        # elif prob >= 0.5 and not self.is_test and pos_src_item is not None:
        #     pos_example = example.copy()
        #     pos_example["source"] = pos_src_item
        #     pos_example["target"] = torch.cat([self.pos_tgt_item, self.eos_item])
        #     pos_example["prev_output_tokens"] = torch.cat([self.bos_item, self.pos_tgt_item])
        #     examples.append(pos_example)
        # elif not self.is_test and neg_src_item is not None:
        #     neg_example = example.copy()
        #     neg_example["source"] = neg_src_item
        #     neg_example["target"] = torch.cat([self.neg_tgt_item, self.eos_item])
        #     neg_example["prev_output_tokens"] = torch.cat([self.bos_item, self.neg_tgt_item])
        #     examples.append(neg_example)
        # return examples
        item = self.dataset[index]
        # uniq_id, base64_str, text, region_coord = item
        uniq_id, img_id,img_path,caption, text, region_coord, dataset_name, type = item
        image = read_img_general(img_path).convert("RGB")
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])


        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region
        }
        return example
