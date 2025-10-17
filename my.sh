CUDA_VISIBLE_DEVICES=0 python -m train --cfg configs/debug.yaml --cfg_assets configs/assets.yaml

CUDA_VISIBLE_DEVICES=0 nohup python -m train --cfg configs/config_mld_stylemohm.yaml --cfg_assets configs/assets.yaml --nodebug > train_sty.txt

# humanml3d
python -m train --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=0 nohup python -m train --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml --nodebug > train_den.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --nodebug > train_vae.txt

python -m test --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=1 python -m test --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml

python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 1
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example2.txt --render --replication 1
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/style_prompt.txt --render
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/style_prompt16.txt --render
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/mld0809.txt --render --replication 3
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/tsne_example.txt --render
python demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --task reconstrucion

# most vae
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m train --cfg configs/config_most_vae.yaml --cfg_assets configs/assets.yaml --nodebug > train_vae_0416.txt
CUDA_VISIBLE_DEVICES=5,6 nohup python -m train --cfg configs/config_most_vae.yaml --cfg_assets configs/assets.yaml --nodebug > train_vae.txt
CUDA_VISIBLE_DEVICES=0 python -m train --cfg configs/config_most_vae.yaml --cfg_assets configs/assets.yaml --nodebug

# most diffusion
python -m train --cfg configs/config_most_finetune.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=1,2 python -m train --cfg configs/config_most.yaml --cfg_assets configs/assets.yaml --nodebug 
CUDA_VISIBLE_DEVICES=1,2 nohup python -m train --cfg configs/config_most.yaml --cfg_assets configs/assets.yaml --nodebug > train_most.txt

# most finetune
CUDA_VISIBLE_DEVICES=3,4 python -m train --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=6,7 python -m train --cfg configs/config_most_finetune_TMR.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=1,2 python -m train --cfg configs/config_most_finetune_TMA.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m train --cfg configs/config_most_tmatextencoder_TMA.yaml --cfg_assets configs/assets.yaml --nodebug 
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m train --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml --nodebug  > train_most_onlyfinetune_T2M001.txt

# most test
python -m test --cfg configs/config_most.yaml --cfg_assets configs/assets.yaml

python -m test --cfg configs/config_most_vae.yaml --cfg_assets configs/assets.yaml

python -m test --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml
python -m test --cfg configs/config_most_finetune_TMR.yaml --cfg_assets configs/assets.yaml

python -m test --cfg configs/config_style_classifier.yaml --cfg_assets configs/assets.yaml


# demo
python demo_difusion.py --cfg ./configs/config_most.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 1
python demo_difusion.py --cfg ./configs/config_most5.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render
python demo_difusion.py --cfg ./configs/config_most_tmatextencoder.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render

python demo_difusion.py --cfg ./configs/config_most_finetune_T2M.yaml --cfg_assets ./configs/assets.yaml --render --replication 3
python demo_difusion.py --cfg ./configs/config_most_finetune_T2M.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 1
python demo_difusion.py --cfg ./configs/config_most5_finetune_T2M.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render
python demo_difusion.py --cfg ./configs/config_most_finetune_TMR.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render
python demo_difusion.py --cfg ./configs/config_most_finetune_TMA.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 3
python demo_difusion.py --cfg ./configs/config_most_tmatextencoder_TMA.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render

python demo_difusion.py --cfg ./configs/config_most_vae.yaml --cfg_assets ./configs/assets.yaml --task random_sampling --render

python demo_difusion.py --cfg ./configs/config_most_style_transfer.yaml --cfg_assets ./configs/assets.yaml --task style_transfer --render

python demo_difusion.py --cfg ./configs/config_most_finetune_T2M.yaml --cfg_assets ./configs/assets.yaml --task reconstrucion


--replication 3

/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=./results/most/most/samples_2024-03-11-11-35-33 --mode=video --joint_type=HumanML3D

# style classifier
CUDA_VISIBLE_DEVICES=1 python -m train --cfg configs/sra/config_style_classifier.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=3 python -m test --cfg configs/sra/config_style_classifier.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=1 python -m train --cfg configs/sra/config_style_classifier_sra.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=3 python -m test --cfg configs/sra/config_style_classifier_sra.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=4 python -m train --cfg configs/sra/config_style_classifier_100.yaml --cfg_assets configs/assets.yaml --nodebug

python save_classifier_model.py --cfg ./configs/config_style_classifier.yaml --cfg_assets ./configs/assets.yaml


# tensorboard
conda activate mdm
tensorboard --logdir=experiments/mld/
tensorboard --logdir=experiments/most/


# visualize
/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=./results/mld/my_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2024-03-01-19-29-55/Example_100_batch0_1.npy --mode=video --joint_type=HumanML3D

python -m fit --files results/mld/my_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2024-03-01-19-29-55/Example_100_batch0_2.npy --save_folder ./blender_save --cuda True

/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=./results/mld/1222_mld_humanml3d_FID041/samples_2024-04-17-16-08-16/Example_100_batch0_4.npy --mode=video --joint_type=HumanML3D
/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=./results/most/most4_finetune_T2M001/0_Teapot+text/Teapot+a_person_is_playing_basketball._rep0_mesh.npy --mode=video --joint_type=HumanML3D
/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=./results/most/most4_finetune_T2M001/2400_ArmsAboveHead+text/ArmsAboveHead+A_person_is_doing_squats_rep0.npy --mode=video --joint_type=HumanML3D
/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=./results/most/most4_finetune_T2M001/2400_ArmsAboveHead+text/ArmsAboveHead+a_person_is_dancing_rep0.npy --mode=sequence --joint_type=HumanML3D
/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=./results/most/most4_finetune_T2M001/2400_ArmsAboveHead+text/ArmsAboveHead+A_person_is_doing_squats_rep0.npy --mode=sequence --joint_type=HumanML3D

python -m fit --files ./results/most/most4_finetune_T2M001/0_Teapot+text/Teapot+a_person_is_playing_basketball._rep0.npy --save_folder ./blender_save --cuda True

/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=./results/mld/my_StyleMoHM1/samples_2024-03-08-14-35-40 --mode=video --joint_type=HumanML3D


# plot_clusters
python -m scripts.plot_clusters --cfg ./configs/config_most_finetune_T2M.yaml --nodebug --visualize


python -m scripts.tsne_most --cfg configs/config_most_finetune_T2M.yaml --visualize
python -m scripts.tsne_most --cfg configs/config_most_tmatextencoder_TMA.yaml --visualize
python -m scripts.tsne_mld --cfg configs/config_mld_humanml3d.yaml --visualize
python -m scripts.tsne_mld_xiabfa --cfg configs/config_mld_humanml3d.yaml --visualize

python -m scripts.test2_most --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml



# XiaBFA
python -m train --cfg configs/config_most_xiabfa.yaml --cfg_assets configs/assets.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m train --cfg configs/config_most_xiabfa.yaml --cfg_assets configs/assets.yaml --nodebug > train_xiabfa.txt

CUDA_VISIBLE_DEVICES=1,2 python -m train --cfg configs/xiabfa/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=5,6 python -m train --cfg configs/xiabfa/config_most_finetune_TMA.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=5,6 python -m train --cfg configs/xiabfa/config_most_finetune_TMR.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=5,6 nohup python -m train --cfg configs/config_most_xiabfa_finetune.yaml --cfg_assets configs/assets.yaml --nodebug > train_most_xiabfa_finetune.txt

CUDA_VISIBLE_DEVICES=5 python -m train --cfg configs/xiabfa/config_style_classifier_xiabfa.yaml --cfg_assets configs/assets.yaml --nodebug

python -m test --cfg configs/config_most_xiabfa_finetune.yaml --cfg_assets configs/assets.yaml


python demo_difusion.py --cfg ./configs/xiabfa/config_most_xiabfa.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --style Zombie
python demo_difusion.py --cfg ./configs/xiabfa/config_most_finetune_T2M.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --style Zombie


#对比实验
# testset1
python -m scripts.test1 --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/config_most.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/config_most_finetune_TMR.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/config_most_finetune_TMA.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/config_most_tmatextencoder_TMA.yaml --cfg_assets configs/assets.yaml

python -m scripts.test1 --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/xiabfa/config_most_xiabfa.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/xiabfa/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/xiabfa/config_most_finetune_TMA.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1 --cfg configs/xiabfa/config_most_finetune_TMR.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1_baseline2_gen --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml
python -m scripts.test1_baseline2 --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml

python -m scripts.tsne_mld --cfg configs/config_mld_humanml3d.yaml --visualize
python -m scripts.tsne_mld_xiabfa --cfg configs/config_mld_humanml3d.yaml --visualize

python -m scripts.tsne_most --cfg configs/config_most_finetune_T2M.yaml --visualize
python -m scripts.tsne_most --cfg configs/config_most_finetune_TMA.yaml --visualize
python -m scripts.tsne_most --cfg configs/config_most_finetune_TMR.yaml --visualize
python -m scripts.tsne_most --cfg configs/config_most_tmatextencoder_TMA.yaml --visualize

python -m scripts.tsne_most_xiabfa --cfg configs/xiabfa/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml --visualize
python -m scripts.tsne_most_xiabfa --cfg configs/xiabfa/config_most_finetune_TMA.yaml --cfg_assets configs/assets.yaml --visualize
python -m scripts.tsne_baseline2

python -m scripts.test_style_classifier --cfg ./configs/sra/config_style_classifier_100.yaml --cfg_assets ./configs/assets.yaml
python -m scripts.test_style_classifier --cfg ./configs/sra/config_style_classifier_sra.yaml --cfg_assets ./configs/assets.yaml
CUDA_VISIBLE_DEVICES=2 python -m scripts.test_style_classifier --cfg ./configs/sra/config_style_classifier_100.yaml --cfg_assets ./configs/assets.yaml
python -m scripts.test_style_classifier --cfg ./configs/xiabfa/config_style_classifier_xiabfa.yaml --cfg_assets ./configs/assets.yaml


# interpolate
python demo_difusion.py --cfg ./configs/config_most_finetune_T2M.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 3 --task interpolate --style1 ArmsFolded --style2 FlickLegs --alpha 0.5
python demo_difusion.py --cfg ./configs/config_most_finetune_T2M_cla.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 1 --task interpolate --style1 ArmsFolded --style2 LeanRight --alpha 0.5
python demo_difusion.py --cfg ./configs/config_most_finetune_T2M_cla.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 1 --task interpolate --style1 OnPhoneLeft --style2 OnToesBentForward --alpha 0.0
python demo_difusion.py --cfg ./configs/config_most_finetune_T2M_cla.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 1 --task interpolate --style1 Star --style2 DuckFoot --alpha 0.5


# w/o denoise loss 0101
# 100style
CUDA_VISIBLE_DEVICES=4,5 python -m train --cfg configs/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=3 python -m scripts.test1 --cfg configs/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=7 python -m scripts.tsne_most_xiabfa --cfg configs/config_most_finetune_T2M_cla_wodenoise.yaml --visualize
CUDA_VISIBLE_DEVICES=1 python demo_difusion.py --cfg ./configs/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 5 --style Star
# xiabfa
CUDA_VISIBLE_DEVICES=3,4 python -m train --cfg configs/xiabfa/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=4 python -m scripts.test1 --cfg configs/xiabfa/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=7 python -m scripts.tsne_most_xiabfa --cfg configs/xiabfa/config_most_finetune_T2M_cla_wodenoise.yaml --visualize
CUDA_VISIBLE_DEVICES=1 python demo_difusion.py --cfg ./configs/xiabfa/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --render --replication 1 --style Zombie
# loss components
CUDA_VISIBLE_DEVICES=0,7 python -m train --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=0 python -m scripts.test1 --cfg configs/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=0,7 python -m train --cfg configs/config_most_finetune_cla.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=0 python -m scripts.test1 --cfg configs/config_most_finetune_cla.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=0,7 python -m train --cfg configs/xiabfa/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=0 python -m scripts.test1 --cfg configs/xiabfa/config_most_finetune_T2M.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=5,6 python -m train --cfg configs/xiabfa/config_most_finetune_cla.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=7 python -m scripts.test1 --cfg configs/xiabfa/config_most_finetune_cla.yaml --cfg_assets configs/assets.yaml

IBLE_DEVICES=7 python -m scripts.test1 --cfg configs/xiabfa/config_most_AFS_frombegin.yaml --cfg_assets configs/assets.yaml



python demo_difusion_style_transfer.py --cfg ./configs/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets ./configs/assets.yaml --task style_transfer --render

# 1016

CUDA_VISIBLE_DEVICES=0 python -m train --cfg configs/config_difu_vae.yaml --cfg_assets configs/assets.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python -m train --cfg configs/config_difu_diffusion_1KDS.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=4,5 python -m train --cfg configs/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m train --cfg configs/bfa/config_difu_diffusion_1KDS.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=4,5 python -m train --cfg configs/bfa/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml


# style classifier
CUDA_VISIBLE_DEVICES=1 python -m train --cfg configs/config_style_classifier_100.yaml --cfg_assets configs/assets.yaml --nodebug
CUDA_VISIBLE_DEVICES=1 python -m train --cfg configs/bfa/config_style_classifier_16.yaml --cfg_assets configs/assets.yaml --nodebug

# demo
python demo_difusion.py --cfg ./configs/config_difu_diffusion_2AFS.yaml --cfg_assets ./configs/assets.yaml --render --style Old --replication 1

python demo_difusion_style_transfer.py --cfg ./configs/config_most_finetune_T2M_cla_wodenoise.yaml --cfg_assets ./configs/assets.yaml --task style_transfer --render

# test
CUDA_VISIBLE_DEVICES=7 python -m test --cfg configs/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml
CUDA_VISIBLE_DEVICES=7 python -m test --cfg configs/bfa/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml

/nfs/Software/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=./results/mld/my_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2024-03-01-19-29-55/Example_100_batch0_1.npy --mode=video --joint_type=HumanML3D
