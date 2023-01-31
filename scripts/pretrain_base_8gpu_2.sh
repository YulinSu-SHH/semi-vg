export MASTER_PORT=1086
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=4

# clusters
worker_cnt=4
gpus_per_node=4
memory=400000
cpu=4000

# data
batch_size=8

selected_cols=0,1,2,3,4,5,6,7
text_selected_cols=0,1
image_selected_cols=0,1,2
detection_selected_cols=0,1,2
max_src_length=80
max_object_length=30
max_tgt_length=30
patch_image_size=256
max_image_size=512
sample_patch_num=-1
num_bins=1000

# optimization
lr=2e-5
# lr=5e-4
# lr_end=2e-5
lr_end=5e-06
clip_grad=5.0
schedule='polynomial_decay'
label_smoothing=0.0
weight_decay=0.01

task=pretrain


init_method="load_pretrain_official"
# init_method="random"
load=/mnt/lustre/share_data/suyulin/hf_ofa_tiny
# pt_model=/mnt/lustre/share_data/suyulin/bart_large.pt
# pt_model=/mnt/lustre/share_data/suyulin/ofa_medium_plaintext.pt
pt_model=/mnt/lustre/share_data/suyulin/bart_base.pt
data_dir=/mnt/lustre/share_data/suyulin/VG/
DATA=${data_dir}/refcocog/refcocog_label.tsv,${data_dir}/refcocog/refcocog_unlabel.tsv,${data_dir}/refcocog/refcocog_test_u.tsv
jobname=offical-large
log_dir=/mnt/lustre/suyulin/debug/ofa-hf-master/log/${jobname}
save_dir=$log_dir
mkdir -p $log_dir
log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
mkdir -p $save_path
image_root=${data_dir}/scratch
eval_cider_cached=/mnt/lustre/share_data/suyulin/caption_data/cider_cached_tokens/coco-valid-words.p
# save
student_model_config="ofa-base"
save=/mnt/lustre/share_data/suyulin/save_model/${jobname}/
ckpt_frequency=1

# pseudo
pseudo_out_root=${data_dir}/scratch
mkdir -p $pseudo_out_root

srun -p rdbp1_v100_32g -n1 -N 1 --gres=gpu:4 --job-name=${jobname} --comment wbsR-SC221736.001 \
python -m torch.distributed.launch --nproc_per_node=${worker_cnt} --master_port=9998 main_train.py \
       --tables=${DATA} \
       --pseudo-out-root=${pseudo_out_root} \
       --image-root=${image_root} \
       --text-data=${text_data} \
       --detection-data=${detection_data} \
       --selected-cols=${selected_cols} \
       --text-selected-cols=${text_selected_cols} \
       --image-selected-cols=${image_selected_cols} \
       --detection-selected-cols=${detection_selected_cols} \
       --neg-sample-dir=${neg_sample_dir} \
       --label-smoothing=${label_smoothing} \
       --batch-size=${batch_size} \
       --max-src-length=${max_src_length} \
       --max-tgt-length=${max_tgt_length} \
       --max-object-length=${max_object_length} \
       --num-bins=${num_bins} \
       --patch-image-size=${patch_image_size} \
       --eval-cider-cached-tokens=${eval_cider_cached} \
       --sample-patch-num=${sample_patch_num} \
       --max-image-size=${max_image_size} \
       --task=${task} \
       --schedule=${schedule} \
       --init-method=${init_method} \
       --student-model-config=${student_model_config} \
       --load=${load} \
       --pt_model=${pt_model} \
       --hf_model_dir=${save} \
       --micro-batch-size=${batch_size} \
       --num-epochs=40 \
       --best-score=10e10 \
       --metric_gc=cider \
       --metric_vg=Acc@0.5 \
       --lr=${lr} \
       --lr-end=${lr_end} \
       --do-train\
       --do-predict\
       --ckpt-frequency=${ckpt_frequency} \
       --weight-decay=${weight_decay} \
       --clip-grad=${clip_grad} \
       --output-dir=${save} \
       --worker-cnt=${worker_cnt} \
       --gpus-per-node=${gpus_per_node} > ${log_file} 2>&1
