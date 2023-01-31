predictory="zz1424:s3://visual_grounding/events/data/"
model='/mnt/lustre/share_data/zhangzhao2/VG/OFA_official_ckpt/hf_ofa_base'
tokenizer_path='/mnt/cache/zhangzhao2/codes/ofa-hf/tokenizer'
save_dir='/mnt/lustre/share_data/zhangzhao2/VG/log/hf-ofa-base/events'
beam=5

echo "Model checkpoint is ${model}"
echo "Directory to save is ${save_dir}"
mkdir -p $save_dir

echo "refclef"
ref="null ref"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/OFA_ceph_annos/ofa_eval_vg_refclef.jsonl'
echo "Run ofa visual grounding ${ref} model ${model} "
srun -p stc1_v100_32g -n1 -N 1 --gres=gpu:1 --job-name=ofa_hf_eval_pretrain_on_event --comment wbsR-SC221419.001 \
python ofa-base_vg_inference.py \
--predictory ${predictory} \
--model ${model} \
--tokenizer_path=${tokenizer_path} \
--ref="${ref}" \
--beam=${beam} \
--val_dir ${val_dir} \
--save_dir ${save_dir} \
--gpu 0 &