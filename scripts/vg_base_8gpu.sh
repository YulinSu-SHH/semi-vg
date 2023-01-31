export MASTER_PORT=1086
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

# clusters
worker_cnt=8
gpus_per_node=8
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
lr_end=5e-06
clip_grad=5.0
schedule='polynomial_decay'
label_smoothing=0.0
weight_decay=0.01

task=refcocog


init_method="load_pretrain"
# load="/mnt/lustre/suyulin/debug/ofa-hf-master/save_model/hf-ofa-base-semi/pretrain/1206_2247/gs9746"
load="/mnt/lustre/share_data/suyulin/hf_ofa_tiny"

# data_dir=/mnt/lustre/share_data/zhangzhao2/VG/OFA_ceph_annos
# neg_sample_dir=/mnt/lustre/suyulin/debug/ofa-hf-master/negative_sample
# DATA=/mnt/lustre/suyulin/debug/ofa-hf-master/ofa_p_vl_0_new.tsv
# text_data=/mnt/lustre/suyulin/debug/ofa-hf-master/ofa_p_mlm_pile_new.tsv
# detection_data=/mnt/lustre/suyulin/debug/ofa-hf-master/ofa_p_det_coco_lvis_obj365_oidet_clean_new.tsv
data_dir=/mnt/lustre/share_data/suyulin/VG/
DATA=${data_dir}/refcocog/refcocog_train.tsv,${data_dir}/refcocog/refcocog_unlabel.tsv,${data_dir}/refcocog/refcocog_test_u.tsv
jobname=vg-tiny-pretrain
# jobname=hf-ofa-base-semi-retrainfrom21epoch
log_dir=/mnt/lustre/suyulin/debug/ofa-hf-master/log/${jobname}
save_dir=${log_dir}
mkdir -p $log_dir
log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
mkdir -p $save_path

# save
student_model_config="ofa-tiny"
save=/mnt/lustre/share_data/suyulin/save_model/${jobname}/
ckpt_frequency=2

srun -p rdbp1_v100_16g -n1 -N 1 --gres=gpu:8 --job-name=${jobname} --comment wbsR-SC221419.001 \
python -m torch.distributed.launch --nproc_per_node=${worker_cnt} --master_port=9998 main_train.py \
       --tables=${DATA} \
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
       --sample-patch-num=${sample_patch_num} \
       --max-image-size=${max_image_size} \
       --task=${task} \
       --schedule=${schedule} \
       --init-method=${init_method} \
       --student-model-config=${student_model_config} \
       --load=${load} \
       --micro-batch-size=${batch_size} \
       --num-epochs=40 \
       --best-score=10e10 \
       --metric=loss \
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
