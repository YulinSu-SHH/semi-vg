ref="dog without leash"
predictory="zz1424:s3://visual_grounding/events/data/"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/dog/val.jsonl'
save_dir='/mnt/lustre/share_data/zhangzhao2/VG/log/hf-ofa-base-cc/event/dog/gs79708'
model='/mnt/lustre/share_data/zhangzhao2/VG/log/hf-ofa-base-cc/pretrain/1009_2221/gs79708'

echo "Run ofa visual grounding ${ref} model ${model} "
srun -p stc1_v100_32g -n1 -N 1 --gres=gpu:1 --job-name=ofa_hf_eval_pretrain_on_event --comment wbsR-SC221419.001 \
python ofa-base_vg_inference.py \
--predictory ${predictory} \
--model ${model} \
--ref="${ref}" \
--val_dir ${val_dir} \
--save_dir ${save_dir} \
--gpu 0