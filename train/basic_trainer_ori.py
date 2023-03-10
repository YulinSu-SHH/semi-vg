import torch
from collections import abc
from tqdm import tqdm
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
import os
import logging
from textbrewer.compatibility import mask_dtype, is_apex_available
from textbrewer.utils import cycle
from textbrewer.distiller_utils import move_to_device
import pdb
from generate import sequence_generator
from data_utils.ofa_dataset import collate_fn
has_apex = is_apex_available()
if has_apex:
    from apex import amp


logger = logging.getLogger("Distillation")

class no_op:
    @staticmethod
    def add_scalar(*args, **kwargs):
        pass

def auto_forward(model, batch, args):
    if type(batch) is dict:
        if isinstance(model, (list, tuple)):
            results = [v(**batch, **args) for v in model]
        elif isinstance(model, dict):
            results = {k: v(**batch, **args) for k, v in model.items()}
        else:
            results = model(**batch, **args)
    else:
        if isinstance(model, (list, tuple)):
            results = [v(*batch, **args) for v in model]
        elif isinstance(model, dict):
            results = {k: v(*batch, **args) for k, v in model.items()}
        else:
            results = model(*batch, **args)
    return results


def get_outputs_from_batch(batch, device, model, args):
    # batch = move_to_device(batch, device)
    # pdb.set_trace()
    # results = model(**batch["net_input"], **args)  # ({yes}, {})
    # results = auto_forward(model, batch, args)

    # results = model(**batch[0]["net_input"], **args)  # ({yes}, {})
    # results = model(*batch["net_input"], *args)

    if isinstance(batch, (list, tuple)):
        batch[0] = move_to_device(batch[0], device)
        batch[1] = move_to_device(batch[1], device)
        batch_v1 = batch[0]["net_input"]
        batch_v2 = batch[1]["net_input"]
        results = [model(**batch_v1, **args), model(**batch_v2, **args)]
    else:
        batch = move_to_device(batch, device)
        results = model(**batch["net_input"], **args)

    # pdb.set_trace()

    return batch, results



class BasicTrainer(object):
    """
        It performs supervised training, not distillation. It can be used for training the teacher model.

        Args:
            train_config (:class:`TrainingConfig`): training configuration.
            model (:class:`torch.nn.Module`): model to be trained.
            adaptor (Callable)???adaptor of the model.

        The role of `adaptor` is explained in :py:func:`adaptor`.
    """
    def __init__(self,train_config,model,adaptor,args):
        self.t_config = train_config
        self.model = model
        self.adaptor = adaptor
        self.local_rank = self.t_config.local_rank
        self.rank = 0
        if self.local_rank != -1:
            self.rank = torch.distributed.get_rank()
        if self.t_config.log_dir is not None and self.rank == 0:
            self.tb_writer = SummaryWriter(log_dir=self.t_config.log_dir)
        else:
            self.tb_writer = no_op
        self.print_freq = 2
        self.tokenizer = args.tokenizer
        self.args=args
        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.remove_grounded_captioning = args.remove_grounded_captioning = False
        self.remove_visual_grounding = args.remove_visual_grounding = False

    def save_and_callback(self, global_step, step, epoch, callback):
        if self.rank != 0:
            torch.distributed.barrier()  # save and eval with single process
        else:
            logger.info(
                f"Saving at global step {global_step}, epoch step {step + 1} epoch {epoch + 1}"
            )
            # coreModel = self.model.module if hasattr(
            #     self.model, "module") else self.model
            # state_dict = coreModel.state_dict()
            # torch.save(
            #     state_dict,
            #     os.path.join(self.t_config.output_dir, f"gs{global_step}.pkl"))
            self.model.module.save_pretrained(os.path.join(self.t_config.output_dir, f"gs{global_step}"))
            
            if self.local_rank == 0:
                torch.distributed.barrier()
        if callback is not None:
            logger.info("Running callback function...")
            callback(model=self.model, step=global_step)
            self.model.train()

    def initialize_training(self, optimizer, scheduler_class, scheduler_args,
                            scheduler):

        logger.debug("Optimizer param group: ")
        logger.debug(
            f"{[[s.shape for s in g['params']] for g in optimizer.param_groups]}"
        )

        # update scheduler
        if scheduler_class is not None:
            # overwrite scheduler
            scheduler = scheduler_class(**{'optimizer': optimizer},
                                        **scheduler_args)

        if self.t_config.fp16:
            if not has_apex:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            self.model, optimizer = amp.initialize(
                    self.model,
                    optimizer,
                    opt_level=self.t_config.fp16_opt_level)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
        elif self.t_config.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        tqdm_disable = None if self.rank == 0 else True
        return optimizer, scheduler, tqdm_disable

    def train_on_batch(self, batch, args):
        batch, results = get_outputs_from_batch(
            batch, self.t_config.device, self.model, args)
        results = self.adaptor(batch, results)
        if results["sample_size"] >= 1:
            total_loss = results["losses"] / results["sample_size"]
        else:
            total_loss = results["losses"]
        return total_loss

    def write_loss(self, total_loss, writer_step, losses_dict=None):
        if self.rank == 0:
            cpu_total_loss = total_loss.cpu().item()
            self.tb_writer.add_scalar('scalar/total_loss', cpu_total_loss,
                                      writer_step)
            if losses_dict is not None:
                for name, loss in losses_dict.items():
                    cpu_loss = loss.cpu().item()
                    self.tb_writer.add_scalar(f"scalar/{name}", cpu_loss,
                                              writer_step)


    def train_with_num_steps(self, optimizer, scheduler, tqdm_disable,
                             dataloader, max_grad_norm, num_steps, callback,
                             batch_postprocessor, **args):
        total_global_steps = num_steps
        ckpt_steps = int(self.t_config.ckpt_steps)
        num_steps = int(num_steps)
        print_every = ckpt_steps // self.print_freq
        if print_every == 0:
            print_every = ckpt_steps
        checkpoints = [
            i * ckpt_steps for i in range(1, num_steps // ckpt_steps + 1)
        ] + [total_global_steps]
        logger.info(f"Total training steps: {total_global_steps}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0
        for step, batch in tqdm(enumerate(cycle(dataloader)),
                                disable=tqdm_disable):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            total_loss, losses_dict = self.train_on_batch(batch, args)

            self.write_loss(total_loss, writer_step, losses_dict)
            writer_step += 1

            total_loss /= self.t_config.gradient_accumulation_steps
            if self.t_config.fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            if (step + 1) % self.t_config.gradient_accumulation_steps == 0:
                if max_grad_norm > 0:
                    if self.t_config.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if (global_step) % print_every == 0:
                    logger.info(
                        f"Global step: {global_step}, epoch step:{step+1}")
                if (global_step % ckpt_steps
                        == 0) or global_step == total_global_steps:
                    self.save_and_callback(global_step, step, 0, callback)
                    loss_dict_str = ", ".join(
                        [f"{k}: {v.item()}" for k, v in losses_dict.items()])
                    logger.info(f"Loss dict: {loss_dict_str}")
            if global_step >= total_global_steps:
                logger.info("Training finished")
                return

    def train_with_num_epochs(self, optimizer, scheduler, tqdm_disable,
                              dataloader, max_grad_norm, num_epochs, callback,
                              batch_postprocessor, **args):

        if isinstance(dataloader, list):
            train_steps_per_epoch = sum([len(
                dataloader[i]) for i in range(len(dataloader))]) // len(dataloader) // self.t_config.gradient_accumulation_steps
        else:
            train_steps_per_epoch = len(
                dataloader) // self.t_config.gradient_accumulation_steps
        total_global_steps = train_steps_per_epoch * num_epochs
        print_every = train_steps_per_epoch // self.print_freq
        print_every = 500
        if print_every == 0:
            print_every = train_steps_per_epoch
        checkpoints=[i for i in range(0,train_steps_per_epoch,100)]
        # checkpoints = [
        #     int(train_steps_per_epoch * ci / self.t_config.ckpt_frequency)
        #     for ci in range(self.t_config.ckpt_frequency)
        # ]
        logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0

        for current_epoch in tqdm(range(int(num_epochs)),
                                  disable=tqdm_disable):
            loader_label,loader_unlabel=dataloader
            if self.local_rank != -1 and hasattr(loader_label, 'sampler'):
                loader_label.sampler.set_epoch(
                    current_epoch
                )  #In distributed mode, calling the set_epoch method is needed to make shuffling work;
            if self.local_rank != -1 and hasattr(loader_unlabel, 'sampler'):
                loader_unlabel.sampler.set_epoch(
                    current_epoch
                ) 
            logger.info(f"Epoch {current_epoch+1}")
            optimizer.zero_grad()
            logger.info(
                f"Length of current epoch in forward batch: {len(loader_label)}  {len(loader_unlabel)}")
            loader_unlabel_iterator = iter(loader_unlabel)
            for step, batch in tqdm(enumerate(loader_label),disable=tqdm_disable):
                try:
                    batch_unlabel = next(loader_unlabel_iterator)
                except StopIteration:
                    loader_unlabel_iterator = iter(loader_unlabel)
                    batch_unlabel = next(loader_unlabel_iterator)
                batch = batch_postprocessor(batch)
                batch_unlabel=batch_postprocessor(batch_unlabel)
                batch_unlabel_task1=[]
                batch_unlabel_task2=[]
                losses_dict={}

                # train on labeled data

                total_loss = self.train_on_batch(batch, args)
                losses_dict["loss_label"]=total_loss
                if current_epoch>=50:
                    # train on unlabeled data
                    generator = sequence_generator.SequenceGenerator(
                        tokenizer=self.tokenizer,
                        beam_size=self.args.beam,
                        max_len_b=self.args.max_len_b,
                        min_len=self.args.min_len,
                        no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                    )
                    batch_unlabel = move_to_device(batch_unlabel[0], self.t_config.device)
                    data = batch_unlabel["net_input"]
                    with torch.no_grad():
                        try:
                            gen_output_ori = generator.generate([self.model], data)
                        except:
                            gen_output = [gen_output_ori[i][0]['tokens'] for i in range(len(gen_output_ori))] 
                        for i in range(len(gen_output)):
                            res = self.tokenizer.batch_decode(gen_output[i][:-1], skip_special_tokens=True)
                            out_bbox_list= [' '.join(res[j: j+4]) for j in range(0, len(res), 4) if j+4 <= len(res)]
                            for out_bbox in out_bbox_list:
                                coord_list = self.bin2coord(f'{out_bbox}', batch_unlabel['w_resize_ratios'][i], batch_unlabel['h_resize_ratios'][i])
                                x0,y0,x1,y1=coord_list
                                

                    for idx in range(len(batch_unlabel)):
                        batch_unlabel[0] = move_to_device(batch_unlabel[0], self.t_config.device)
                        batch_v1 = batch_unlabel[0]
                        inputs = batch_v1["net_input"]["input_ids"][idx].unsqueeze(0)
                        patch_image=batch_v1["net_input"]["patch_images"][idx].unsqueeze(0)
                        patch_mask=batch_v1["net_input"]["patch_masks"][idx].unsqueeze(0)
                        
                        data = {}
                        data["net_input"] = {"input_ids": inputs.to(self.t_config.device), 'patch_images': patch_image, 'patch_masks':patch_mask}
                        with torch.no_grad():
                            try:
                                gen_output = generator.generate([self.model.module], data)
                            except:
                                gen_output = generator.generate([self.model], data)
                        gen_coord = gen_output[0][0]
                        res_coord = self.tokenizer.batch_decode(gen_coord['tokens'], skip_special_tokens=True)
                        out_bbox_list = [' '.join(res_coord[i: i+4]) for i in range(0, len(res_coord), 4) if i+4 <= len(res_coord)]
                        tgt_item_v1=batch_v1['refs'][idx].cpu()
                        region_prefix_item = self.tokenizer('  what does the region describe? region:', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                        
                        for out_bbox in out_bbox_list:
                            coord_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(out_bbox.split()))
                            
                            src_item_v1 = torch.cat([region_prefix_item, coord_item])
                            example_v1 = {
                                "id": batch_v1["id"][idx],
                                "source": torch.cat([self.bos_item, src_item_v1, self.eos_item]),
                                "patch_image": batch_v1["net_input"]["patch_images"][idx],
                                "patch_mask": patch_mask,
                                # "code_mask": torch.tensor([True]),
                                "target": torch.cat([tgt_item_v1,self.eos_item]),
                                "prev_output_tokens": torch.cat([self.bos_item, tgt_item_v1]),
                                "conf": torch.tensor([0.0]) if self.remove_grounded_captioning else torch.tensor([1.0]),
                            }
                            batch_unlabel_task1.append(example_v1)

                        ### Task2 ###
                        batch_unlabel[1] = move_to_device(batch_unlabel[1], self.t_config.device)
                        batch_v2 = batch_unlabel[1]
                        inputs = batch_v2["net_input"]["input_ids"][idx].unsqueeze(0)
                        patch_image=batch_v2["net_input"]["patch_images"][idx].unsqueeze(0)
                        patch_mask=batch_v2["net_input"]["patch_masks"][idx].unsqueeze(0)
                        data = {}
                        data["net_input"] = {"input_ids": inputs.to(self.t_config.device), 'patch_images': patch_image, 'patch_masks':patch_mask}
                        with torch.no_grad():
                            gen_output = generator.generate([self.model], data)

                        gen_ref = gen_output[0][0]
                        res_ref = self.tokenizer.batch_decode(gen_ref['tokens'], skip_special_tokens=True)
                        src_caption= ''.join(res_ref[:-1])  
                        
                        tgt_item_v2 = batch_v2['refs'][idx].cpu()
                        src_item_v2 = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt",
                                            add_special_tokens=False).input_ids.squeeze(0)
                        
                        vg_example = {
                            "id": batch_v2["id"][idx],
                            "source": torch.cat([self.bos_item, src_item_v2, self.eos_item]),
                            "patch_image": batch_v2["net_input"]["patch_images"][idx],
                            "patch_mask": patch_mask,
                            "target": torch.cat([ tgt_item_v2,self.eos_item]),
                            "prev_output_tokens": torch.cat([self.bos_item, tgt_item_v2]),
                            "conf": torch.tensor([1.0])
                            }
                        batch_unlabel_task2.append(vg_example)
                        
                    batch_unlabel_task1=collate_fn(batch_unlabel_task1, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
                    batch_unlabel_task2=collate_fn(batch_unlabel_task2, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
                    loss_coords_reconstruction = self.train_on_batch(batch_unlabel_task1, args)
                    loss_refs_reconstruction = self.train_on_batch(batch_unlabel_task2, args)
            
                    total_loss += loss_coords_reconstruction + loss_refs_reconstruction 
                    losses_dict['loss_coords_reconstruction'] = loss_coords_reconstruction
                    losses_dict['loss_refs_reconstruction'] = loss_refs_reconstruction
                # losses_dict['loss_imgs_reconstruction'] = loss_imgs_reconstruction

                self.write_loss(total_loss, writer_step, losses_dict)
                writer_step += 1

                total_loss /= self.t_config.gradient_accumulation_steps
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                if (step + 1) % self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (global_step - 1) % print_every == 0:
                        logger.info(
                            f"Global step: {global_step}, epoch step:{step+1}")
                        logger.info(f"Loss dict: {losses_dict}")
                    if (global_step % train_steps_per_epoch in checkpoints) \
                            and ((current_epoch+1) % self.t_config.ckpt_epoch_frequency==0 or current_epoch+1 == num_epochs):
                        losses_str_dict = {
                            k: v.item()
                            for k, v in losses_dict.items()
                        }
                        logger.info(f"Loss dict: {losses_str_dict}")
                        self.save_and_callback(global_step, step,
                                               current_epoch, callback)

            logger.info(f"Epoch {current_epoch+1} finished")

    def train(self,
              optimizer,
              dataloader,
              num_epochs=None,
              scheduler_class=None,
              scheduler_args=None,
              scheduler=None,
              max_grad_norm=-1.0,
              num_steps=None,
              callback=None,
              batch_postprocessor=None,
              **args):
        """
        trains the model.

        Args:
            optimizer: optimizer.
            dataloader: dataset iterator.
            num_epochs (int): number of training epochs.
            num_steps (int): number of training steps. If it is not None, distiller will ignore `num_epochs` and trains for `num_steps`, and dataloader can have an unkonwn size, i.e., has no `__len__` attribute. Dataloader will be cycled automatically after iterating over the whole dataset.
            callback (Callable): function called after each epoch, can be None. It is called as ``callback(model=self.model_S, step = global_step)``. It can be used to evaluate the model at each checkpoint.
            batch_postprocessor (Callable): a function for post-processing batches. It should take a batch and return a batch. Its output is fed to the models and adaptors.
            scheduler_class (class): the class of the scheduler to be constructed.
            scheduler_args (dict): arguments (excluding `optimizer`) passed to the `scheduler_class` to construct the scheduler object. See the example below.
            scheduler (deprecated): used to adjust learning rate, optional, can be None, is deprecated in favor of `scheduler_class` and `scheduler_args`.
            max_grad_norm (float): Maximum norm for the gradients (-1 means no clipping). Default: -1.0
            **args: additional arguments fed to the model.
        Note:
            * If the batch is a list or tuple, model is called as: ``model(*batch, **args)``. Make sure the order of elements in the batch matches their order in ``model.forward``.
            * If the batch is a dict, model is called as: ``model(**batch,**args)``. Make sure the keys of the batch match the arguments of the ``model.forward``.
        Note:
            If you want to provide a lr scheduler, DON'T USE `scheduler` , use `scheduler_class` and `scheduler_args` instead. Example:

            .. code-block::

                from transformers import get_linear_schedule_with_warmup
                trainer.train(optimizer, scheduler_class = get_linear_schedule_with_warmup, scheduler_args= {'num_warmup_steps': 100, 'num_training_steps': 1000})
        """
        optimizer, scheduler, tqdm_disable = self.initialize_training(
            optimizer, scheduler_class, scheduler_args, scheduler)

        assert not (num_epochs is None and num_steps is None)
        if num_steps is not None:
            self.train_with_num_steps(optimizer, scheduler, tqdm_disable,
                                      dataloader, max_grad_norm, num_steps,
                                      callback, batch_postprocessor, **args)
        else:
            self.train_with_num_epochs(optimizer, scheduler, tqdm_disable,
                                       dataloader, max_grad_norm, num_epochs,
                                       callback, batch_postprocessor, **args)

    def __enter__(self):
        if isinstance(self.model,(list,tuple)):
            self.model_S_is_training = [model_s.training for model_s in self.model]
            for model_s in self.model:
                model_s.eval()
        elif isinstance(self.model,dict):
            self.model_S_is_training = {name:model.training for name,model in self.model.items()}
            for name in self.model:
                self.model[name].eval()
        else:
            self.model_S_is_training = self.model.training
            self.model.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.model,(list,tuple)):
            for i in range(len(self.model_S_is_training)):
                self.model[i].train(self.model_S_is_training[i])
        elif isinstance(self.model,dict):
            for name, is_training in self.model_S_is_training.items():
                self.model[name].train(is_training)
        else:
            self.model.train(self.model_S_is_training)


