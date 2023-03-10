import os
import numpy as np
import random
import pathlib
import torch
import logging
import pipeline

from arguments import get_args
from datetime import datetime
from pipeline import initialize_distributed
from init_task import build_task
from ofa.modeling_ofa import OFAModel



teacher_model_type = 'bert-base-uncased'
TOKENIZER_PATH = "./tokenizer"
CUR_PATH = pathlib.Path().resolve()


def args_check(args, logger):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu


def main():
    import PIL
    print("PIL version", PIL.__version__)
    print("CUDA device name", torch.cuda.get_device_name(0))
    # ---------- SETTINGS ----------------
    torch.backends.cudnn.enabled = True
    args = get_args()

    cur_time = datetime.strftime(datetime.now(), '%m%d_%H%M')
    if args.postfix is not None:
        args.output_dir = os.path.join(args.output_dir, args.postfix)
    folder_name = args.task + "/" + cur_time
    args.output_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(args.output_dir, exist_ok=True)
    log_output_dir = args.output_dir
    os.makedirs(log_output_dir, exist_ok=True)
    if args.rank == 0:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(log_output_dir, f'log_{args.rank}.txt'))
            ],
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger = logging.getLogger("Main")
        print(logger)
    else:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            handlers=[logging.StreamHandler()],
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger = logging.getLogger("Main")
        print(logger)


    logger.info("---------- SETTINGS ----------------")
    initialize_distributed(args)


    logger.warning(
        f'args.world_size = {args.world_size}, args.rank ={args.rank}, args.local_rank = {args.local_rank}'
    )
    device, n_gpu = args_check(args, logger)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Args : {args.__dict__}")

    logger.info("---------- RANDOM SEEDs ----------------")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    build_task(args)

    # ---------- DATA LOADER ----------------
    logger.info("---------- DATA LOADER ----------------")

    train_loader_label, train_loader_unlabel,val_loader = pipeline.get_data_loader(args, logger)
    logger.info(f'train loader label: {train_loader_label}, length: {len(train_loader_label)}')
    logger.info(f'train loader unlabel: {train_loader_unlabel}, length: {len(train_loader_unlabel)}')
    # if len(args.train_dataset) > 1:
    #     args.num_train_steps = sum([len(
    #         train_loader[i]) for i in range(len(args.train_dataset))]) // args.gradient_accumulation_steps * args.num_epochs
    # else:
    #     args.num_train_steps = len(
    #         train_loader) // args.gradient_accumulation_steps * args.num_epochs
    args.num_train_steps = (len(
        train_loader_label)+len(train_loader_unlabel)) // args.gradient_accumulation_steps * args.num_epochs

    # ---------- MODELs ----------------
    logger.info("----------- MODELS ------------")

    models = []
    model_paths = args.load.split(",")
    for i in range(len(model_paths)):
        logger.info(f'model path of model {i}: {model_paths[i]}')
        if args.generator_version == 'fairseq':
            model = OFAModel.from_pretrained(model_paths[i], use_cache=True)
        else:
            model = OFAModel.from_pretrained(model_paths[i], use_cache=False)
        logger.info(' Number of model {} parameters on rank {}: {}'.format(
            i,
            torch.distributed.get_rank(),
            sum([p.nelement() for p in model.parameters()])))
        model.to(device)
        logger.info(model)
        logger.info(f'Config of model: {model.config.to_dict()}')
        
        # model = torch.nn.parallel.DistributedDataParallel(
        #         model,
        #         device_ids=[args.local_rank],
        #         output_device=args.local_rank,
        #         find_unused_parameters=True,
        #         broadcast_buffers=False
        #     )
        
        models.append(model)
        import evaluation
        args.best_score = 0
        print("args.patch_image_size", args.patch_image_size)
        logger.info(f"Results of Model {model_paths[i]}")
        logger.info("________________________")
        evaluation.ddp_evaluate(model, 0, val_loader, args, logger)
        logger.info(f"The above is the results of Model {model_paths[i]}")


if __name__ == '__main__':
    main()
