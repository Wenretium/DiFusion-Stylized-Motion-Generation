# DiFusion: Flexible Stylized Motion Generation Using Digest-and-Fusion Scheme

### [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11202393) - TVCG 2025

We propose DiFusion, a framework for diversely **stylized motion generation**. It offers flexible control of content through texts and style via multiple modalities, i.e., textual labels or motion sequences. Additionally, our approach can be extended to applications, such as motion style interpolation and motion style transfer. 

<p float="center">
  <img src="checkpoints/method.png"/>
</p>

## 🚩 News
- [2025/10/17] first release, code for both training, evaluation and demo

## ⚙️ Setup
  
### 1. Conda environment

```
conda create python=3.9 --name mld
conda activate mld
```

Install the packages in `requirements.txt` and install [PyTorch 1.12.1](https://pytorch.org/)

```
pip install -r requirements.txt
```

We test our code on Python 3.9.12 and PyTorch 1.12.1.

### 2. Dependencies

Run the script to download dependencies materials:

```
bash prepare/download_smpl_model.sh
bash prepare/prepare_clip.sh
bash prepare/download_t2m_evaluators.sh
```

### 3. Pre-train model

Run the script to download the pre-train model

```
bash prepare/download_pretrained_models.sh
```

</details>

## 🚀 Demo

We support text file or keyboard input, the generated motions are npy files.
Please check the `configs/asset.yaml` for path config, TEST.FOLDER as output folder.

Then, run the following script:

```
python demo_difusion.py --cfg ./configs/config_difu_diffusion_2AFS.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --style Star --render
```

Some parameters:

- `--example=./demo/example.txt`: input file as text prompts
- `--task=textstyle2motion`: default task, text+style -> motion
- `--replication`: generate motions for same inputs multiple times
- `--render`: whether to render the generated motion as an MP4 animation

The outputs:

- `npy file`: the generated motions with the shape of (nframe, 22, 3)
- `text file`: the input text prompt

### Motion Style Interpolate

```
python demo_difusion.py --cfg ./configs/config_difu_diffusion_2AFS.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --task interpolate --style1 ArmsFolded --style2 FlickLegs --alpha 0.5 --render
```

Some parameters:

- `--task=interpolate`: enables stylized motion generation via style interpolation
- `--style1`: the first style used for interpolation
- `--style2`: the second style used for interpolation
- `--alpha`: the interpolation weight between style1 and style2

### Motion Style Transfer

```
python demo_difusion_style_transfer.py --cfg ./configs/config_difu_diffusion_2AFS.yaml --cfg_assets ./configs/assets.yaml --content 000222 --style Old --render
```

- `--content`: the source motion content, identified by its name in HumanML3D (path can be modified in the code)
- `--style`: the target style


## 💻 Train your own models

### 1. Prepare the datasets

#### HumanML3D
Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup.
#### 100STYLE & BFA
Download our processsed datasets from [here](https://drive.google.com/drive/folders/1nyBsX7PJOGyA7UPQX6hZMPJj2IrBsdOa?usp=sharing). 

#### Change datasets paths in `configs/assets.yaml`, e.g.

```
DIFUSION:
    ROOT_T2M: 'path/to/HumanML3D'
    SPLIT_ROOT_T2M: 'path/to/HumanML3D'
    ROOT_S2M: 'path/to/STYLE100'
    SPLIT_ROOT_S2M: 'path/to/STYLE100'
```

### 2.1. Ready to train VAE model

Please first check the parameters in `configs/config_difu_vae.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command:

```
python -m train --cfg configs/config_difu_vae.yaml --cfg_assets configs/assets.yaml --nodebug
```

### 2.2. Ready to train style classifier

Please first check the parameters in `configs/config_style_classifier_100.yaml`, e.g. `NAME`,`DEBUG`.

Then, run the following command:

```
python -m train --cfg configs/config_style_classifier_100.yaml --cfg_assets configs/assets.yaml --nodebug
```

### 2.3. Ready to train MLD model for Knowledge Digest Stage (KDS)

Please update the parameters in `configs/config_difu_diffusion_1KDS.yaml`:
+ `NAME`
+ `TRAIN.PRETRAINED_VAE`, change to your `latest ckpt model path` in step 2.1.

Then, run the following command:

```
python -m train --cfg configs/config_difu_diffusion_1KDS.yaml --cfg_assets configs/assets.yaml --nodebug
```

### 2.4. Ready to train MLD model for Adaptive Fusion Stage (AFS)

Please update the parameters in `configs/config_difu_diffusion_2AFS.yaml`:
+ `NAME`, e.g. `difusion_diffusion_2AFS`
+ `TRAIN.PRETRAINED_CLASSIFIER`, change to your `latest ckpt model path` in step 2.2.
+ `TRAIN.RESUME`, change to your `latest ckpt model path` after KDS in step 2.3.

Then, run the following command:

```
python -m train --cfg configs/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml --nodebug
```

### 3. Evaluate the model

Please first put the trained model checkpoint path to `TEST.CHECKPOINT` in `configs/config_difu_diffusion_2AFS.yaml`.

Then, run the following command:

```
python -m test --cfg configs/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml
```

### 4. Train the model on HumanML3D + BFA (Optional)

Similarly, we provide configuration files in `./configs/bfa`:

```
# train style classifier
python -m test --cfg configs/bfa/config_style_classifier_16.yaml --cfg_assets configs/assets.yaml --nodebug
# train for KDS
python -m test --cfg configs/bfa/config_difu_diffusion_1KDS.yaml --cfg_assets configs/assets.yaml --nodebug
# train for AFS
python -m test --cfg configs/bfa/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml --nodebug
# eval
python -m test --cfg configs/bfa/config_difu_diffusion_2AFS.yaml --cfg_assets configs/assets.yaml
```


## 👀 Visualization


### 1. Set up blender - WIP

Refer to [TEMOS-Rendering motions](https://github.com/Mathux/TEMOS) for blender setup, then install the following dependencies.

```
YOUR_BLENDER_PYTHON_PATH/python -m pip install -r prepare/requirements_render.txt
```

### 2. (Optional) Render stickman

Run the following command using blender:

```
# render all npy files in one folder
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D
# render a specific npy file
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=YOUR_NPY_FILE --mode=video --joint_type=HumanML3D
```

### 2. Create SMPL meshes with:

```
# fit all npy files in one folder
python -m fit --dir YOUR_NPY_FOLDER --save_folder TEMP_PLY_FOLDER --cuda True
# fit a specific npy file
python -m fit --files YOUR_NPY_FILE --save_folder TEMP_PLY_FOLDER --cuda True
```

This outputs:

- `mesh npy file`: the generate SMPL vertices with the shape of (nframe, 6893, 3)
- `ply files`: the ply mesh file for blender or meshlab

### 3. Render SMPL meshes

Run the following command to render SMPL using blender:

```
# render all npy files in one folder
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D
# render a specific npy file
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=YOUR_NPY_FILE --mode=video --joint_type=HumanML3D
```

optional parameters:

- `--mode=video`: render mp4 video
- `--mode=sequence`: render the whole motion in a png image.


## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@article{wang2025difusion,
  title={DiFusion: Flexible Stylized Motion Generation Using Digest-and-Fusion Scheme},
  author={Wang, Yatian and Mo, Haoran and Gao, Chengying},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgments

Thanks to [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MLD](hhttps://github.com/ChenFengYe/motion-latent-diffusion) and [SMooDi](https://github.com/neu-vi/SMooDi), our code is partially borrowing from them.

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
