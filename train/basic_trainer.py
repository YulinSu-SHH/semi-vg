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
from .generate_pseudos import generate_pseudo_loader
from metrics.evaluator import Evaluator
from evaluation import bin2coord
has_apex = is_apex_available()
if has_apex:
    from apex import amp


logger = logging.getLogger("Train")

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
            adaptor (Callable)：adaptor of the model.

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
        self.pseudo_out_root=args.pseudo_out_root
        self.world_size=args.world_size

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
        # print("train result: ",results)
        results = self.adaptor(batch, results)
        if results["sample_size"] >= 1:
            total_loss=results["losses"] / results["sample_size"]
        else:
            total_loss=results["losses"] 
        seperate_loss_v1=results["batch_losses"]/results["batch_size"]
        seperate_loss_v2=results["batch_losses2"]
        return total_loss,seperate_loss_v1

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
        checkpoints = [
            int(train_steps_per_epoch * ci / self.t_config.ckpt_frequency)
            for ci in range(self.t_config.ckpt_frequency)
        ]
        logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0
        evaluator = Evaluator()
        
        for current_epoch in tqdm(range(int(num_epochs)),
                                  disable=tqdm_disable):
            loader_label,loader_unlabel=dataloader
            if self.local_rank != -1 and hasattr(loader_label, 'sampler'):
                loader_label.sampler.set_epoch(current_epoch)  #In distributed mode, calling the set_epoch method is needed to make shuffling work;
            logger.info(f"Epoch {current_epoch+1}")
            optimizer.zero_grad()
            if current_epoch>=50:
                logger.info(f"*** Start Pseudo Generation ***")
                out_tsv_paths=[os.path.join(self.pseudo_out_root,"vg_"+str(current_epoch)+".tsv")]
                out_tsv_paths.append(os.path.join(self.pseudo_out_root,"caption_"+str(current_epoch)+".tsv"))
                loader_unlabel = generate_pseudo_loader(self.world_size,loader_unlabel,self.model,self.args,self.t_config.device,logger,out_tsv_paths)
                loader_unlabel_iterator = iter(loader_unlabel)
                if self.local_rank != -1 and hasattr(loader_unlabel, 'sampler'):
                    loader_unlabel.sampler.set_epoch(current_epoch) 
                logger.info(f"Length of current epoch in forward batch: {len(loader_label)+len(loader_unlabel)}")
            else:
                logger.info(f"Length of current epoch in forward batch: {len(loader_label)}")
            
            for step, batch in tqdm(enumerate(loader_label),disable=tqdm_disable):    
                batch = batch_postprocessor(batch)
                losses_dict={}
                # train on labeled data
                total_loss ,seperate_loss= self.train_on_batch(batch, args)
                
                losses_dict["loss_label"]=total_loss
                if current_epoch>=50:
                    try:
                        batch_unlabel = next(loader_unlabel_iterator)
                    except StopIteration:
                        loader_unlabel_iterator = iter(loader_unlabel)
                        batch_unlabel = next(loader_unlabel_iterator)
                    
                    batch_unlabel=batch_postprocessor(batch_unlabel)
                    predictedBoxList=[]
                    gtBoxList=[]
                    predictedBoxList_xy=[]
                    gtBoxList_xy=[]
                    with torch.no_grad():
                        batch_vg = move_to_device(batch[0], self.t_config.device)
                        gen_output_ori = self.args.generator.generate([self.model.module],batch_vg)
                        gen_output = [gen_output_ori[i][0]['tokens'] for i in range(len(gen_output_ori))]
                        
                        for i in range(len(gen_output)):
                            res = self.tokenizer.batch_decode(gen_output[i][:-1], skip_special_tokens=True) 
                            out_bbox_list= [' '.join(res[j: j+4]) for j in range(0, len(res), 4) if j+4 <= len(res)]          
                            ref =self.tokenizer.batch_decode(batch_vg["target"][i][:-1], skip_special_tokens=True) #sample:  ['<bin_0>', '<bin_0>', '<bin_749>', '<bin_749>']
                            ref=' '.join(i for i in ref)
                            for out_bbox in out_bbox_list:
                                try:
                                    coord_list = bin2coord(f'{out_bbox}', batch_vg['w_resize_ratios'][i], batch_vg['h_resize_ratios'][i])
                                except:
                                    coord_list=[0.,0.,0.,0.]
                                    
                                x0,y0,x1,y1=coord_list
                                w=x1-x0
                                h=y1-y0
                                coord_wh=torch.tensor([x0,y0,w,h]).to(self.t_config.device)
                                coord_xy=torch.tensor([x0,y0,x1,y1]).to(self.t_config.device)
                                ref_coord_list = bin2coord(f'{ref}', batch_vg['w_resize_ratios'][i], batch_vg['h_resize_ratios'][i])
                                ref_x0,ref_y0,ref_x1,ref_y1=ref_coord_list
                                ref_w=ref_x1-ref_x0
                                ref_h=ref_y1-ref_y0
                                ref_coord_wh=torch.tensor([ref_x0,ref_y0,ref_w,ref_h]).to(self.t_config.device)
                                ref_coord_xy=torch.tensor([ref_x0,ref_y0,ref_x1,ref_y1]).to(self.t_config.device)
                                predictedBoxList.append(coord_wh)
                                gtBoxList.append(ref_coord_wh)
                                predictedBoxList_xy.append(coord_xy)
                                gtBoxList_xy.append(ref_coord_xy)              
  

                    iouList = evaluator.compute_iou(predictedBoxList, gtBoxList)
                    iouList_idx_sorted = sorted(range(len(iouList)), key = lambda k : iouList[k],reverse=True) 
                    # batch_unlabel_filter = [batch_vg[i] for i in iouList_idx_sorted[:4]]
                    # batch_unlabel_filter=batch_unlabel[iouList_idx_sorted[:4]]
                    loss_unlabel, seperate_loss = self.train_on_batch(batch_unlabel, args)
                    filter_ratio=int(len(seperate_loss)*0.25)
                    seperate_loss_sorted=[seperate_loss[i] for i in iouList_idx_sorted]
                    filter_loss_unlabel=sum(seperate_loss_sorted[:filter_ratio])
                    # seperate_loss_sorted, idx_sorted = torch.sort(seperate_loss, dim=0)
                    # filter_loss_unlabel=torch.sum(seperate_loss_sorted[:filter_ratio])
                    losses_dict["filter_loss_unlabel"]=filter_loss_unlabel
                    total_loss+=filter_loss_unlabel

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
                        self.save_and_callback(global_step, step,current_epoch, callback)

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


