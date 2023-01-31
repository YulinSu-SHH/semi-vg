import os
from sys import platlibdir
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import pdb
import gradio as gr


# APPs
# from demo.apps.ofa_pretrain_interface import interface as ofa_pt
# from demo.apps.ofa_fintune_interface import interface as ofa_ft
from demo.apps.ofa_attr_interface import interface as ofa_attr
from demo.apps.util import parallel_call



title = "跨模态预训练模型 Demo（事件定位）"
description = "当前支持一个多任务预训练模型和一个在下游事件上Finetune后的模型，点击标签页切换" \
              "支持上传图像或点击页面底部的示例，点击\"Submit\"查看运行结果"
# intro = "<img src='/mnt/cache/zhangzhao2/codes/ofa-stc/demo/pictures/sensetime_logo.png'>" \
        #   "<p style='text-align: center'><a href='https://gitlab.bj.sensetime.com/xiechi/ofa_train' target='_blank'>OFA" \
        #   "Repo</a></p> "

intro = "<h1><font size='10'>基于跨模态方法的端到端长尾事件发现</font></h1>"
split_line = "<HR style='FILTER: alpha(opacity=100,finishopacity=0,style=3)' width='80%' color=#987cb9 SIZE=3>"
pt_examples = [
                ['demo/examples/dog/pos_dog.jpg', 'dog without leash'],
                ['demo/examples/fisherman/pos_fisherman.jpg', 'man fishing'],
                ['demo/examples/trash/pos_trash.jpg', 'manhole cover']
            ]

ft_examples = [
                ['demo/examples/dog/neg_dog.jpg', 'dog without leash'],
                ['demo/examples/dog/pos_dog.jpg', 'dog without leash'],
                ['demo/examples/fisherman/neg_fisherman.jpg', 'man fishing'],
                ['demo/examples/fisherman/pos_fisherman.jpg', 'man fishing'],
                ['demo/examples/trash/neg_trash.jpg', 'overfilled bin'],
                ['demo/examples/trash/pos_trash.jpg', 'overfilled bin']
            ]

# <h1><font size='10'>基于跨模态方法的端到端长尾事件发现</font></h1>

with gr.Blocks() as demo:
    with gr.Column():
        gr.Image("demo/pictures/intro_pic.png").style(
            rounded = [False, False, False, False]
        )
        # gr.HTML(intro)
        # gr.HTML(split_line)
        with gr.Row():
            with gr.Column():
                in_img = gr.inputs.Image(type='pil', label='输入图像')
                with gr.Tab("模型对比"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                """
                                ### OFA原始模型/长尾发现通用模型/长尾发现专才模型的对比结果
                                输入要找的事件后，三个模型的检测结果会展示在下方

                                OFA模型不支持“反事实查询”和“多目标查询”
                                """
                            )
                        with gr.Column():
                            cmp_cap = gr.Textbox(label="你要找...")
                            cmp_submit_btn = gr.Button("提交")

                    with gr.Row():
                        out_img_ori = gr.outputs.Image(type='numpy', label='OFA预测结果')
                        out_img_pt = gr.outputs.Image(type='numpy', label='通用模型结果')
                        out_img_ft = gr.outputs.Image(type='numpy', label='专才模型结果')
                    
                    gr.Examples(ft_examples, inputs=[in_img, cmp_cap])



                with gr.Tab("通用模型"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                """
                                ### 智慧城市场景下长尾事件发现的**通用模型**
                                可以在下方文本框中输入任何搜索的事件

                                点击提交，稍等片刻，模型会在右方输出检测结果
                                """
                            )
                            pt_cap = gr.Textbox(label="你要找...")
                            pt_submit_btn = gr.Button("提交")
                        with gr.Column():
                            out_img = gr.outputs.Image(type='numpy', label='检测结果')
                            out_ood = gr.Textbox(label="OOD Score:")
                    gr.Examples(pt_examples, inputs=[in_img, pt_cap])


                with gr.Tab("专才模型"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                """
                                ### 智慧城市场景下长尾事件发现的**专才模型**
                                此模型已经在特定事件上Finetune

                                可以在下方选择框中选择指定事件

                                点击提交，稍等片刻，模型会在右方输出检测结果
                                """
                            )
                            ft_cap = gr.Dropdown(label="选择需要检测事件",value="the salient object",choices=["motorbike with three people", "man fishing", "person with a cigarrete close to the mouth", "stall on a tricycle", "dog without leash", "broken or fallen tree", "overfilled bin", "hanging clothes or sheets"])
                            ft_submit_btn = gr.Button("提交")
                        with gr.Column():
                            out_img = gr.outputs.Image(type='numpy', label='检测结果')
                            out_ood = gr.Textbox(label="OOD Score:")
                    gr.Examples(ft_examples, inputs=[in_img, ft_cap])

                with gr.Tab("属性功能体验"):
                    with gr.Row():
                            gr.Markdown(
                                """
                                ### 智慧城市场景下跨模态预训练模型的**属性功能**
                                目前仅支持行人属性, 可以在下面的选择特定属性或者自定义属性。点击提交，稍等片刻，模型会在右下方的文本框中会输出属性的结果
                                """
                            )
                    with gr.Row():
                        with gr.Tab("特定属性检测"):
                            can_choices = [
                                "what is the person's gender? male, female, or indecipherable?",
                                "what is the color of this person's mask? white, gray, blue, or black?",
                                "does this person have something on his chest? yes, no, or indecipherable?"
                            ]
                            attr_cap = gr.Dropdown(label="选择你要检测的属性",value="what is on the picture?",choices=can_choices)
                            attr_submit_btn = gr.Button("提交")
                        with gr.Tab("自定义属性检测"):
                            attr_open_cap = gr.Textbox(label="问题? 属性A, 属性B, or 属性C?")
                            attr_open_submit_btn = gr.Button("提交")
                    with gr.Row():
                        out_ans = gr.outputs.Textbox(label='检测结果')

                    # gr.Examples(ft_examples, inputs=[in_img, ft_cap])
    # ofa_cmp = parallel_call(ofa_pt, ofa_pt, ofa_ft)
    # cmp_cap.submit(fn=ofa_cmp, inputs=[in_img, cmp_cap], outputs=[out_img_ori, out_img_pt, out_img_ft])  # for enter
    # cmp_submit_btn.click(fn=ofa_cmp, inputs=[in_img, cmp_cap], outputs=[out_img_ori, out_img_pt, out_img_ft])

                
    # pt_cap.submit(fn=ofa_pt, inputs=[in_img, pt_cap], outputs=[out_img, out_ood])  # for enter
    # pt_submit_btn.click(fn=ofa_pt, inputs=[in_img, pt_cap], outputs=[out_img, out_ood])

    attr_submit_btn.click(fn=ofa_attr, inputs=[in_img, attr_cap], outputs=[out_ans])
    attr_open_submit_btn.click(fn=ofa_attr, inputs=[in_img, attr_open_cap], outputs=[out_ans])

    # ft_submit_btn.click(fn=ofa_ft, inputs=[in_img, ft_cap], outputs=[out_img, out_ood])



demo.launch()
