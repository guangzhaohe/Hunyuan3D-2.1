"""Components are mainly borrowed from Trellis3D (https://github.com/microsoft/TRELLIS)"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/hy3dpaint')
import json
import shutil
import pickle
import hashlib
import subprocess
from typing import *
from functools import partial
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import trimesh
import utils3d
import numpy as np
import open3d as o3d
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig


class ControlNet:
    def __init__(self, variant: str = 'canny'):
        self.type = type
        
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-" + variant, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        )
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def run_controlnet_depth(
        self,
        image: np.ndarray, 
        prompt: List[str],
        seed: int = 144, 
    ):
        # image: HxW depth image
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        
        prompt = [p + ", best quality, realistic, extremely detailed" for p in prompt]

        generator = [torch.Generator(device="cpu").manual_seed(seed) for i in range(len(prompt))]

        output = self.pipe(
            prompt,
            image,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
            generator=generator,
            num_inference_steps=20,
        )
        
        image_gen = output['images']

        return np.array(image_gen)

    def run_controlnet_canny(
        self,
        image: np.ndarray, 
        prompt: List[str],
        seed: int = 42, 
        low_thresh: float = 100, 
        high_thresh: float = 200,
    ):
        # image: H, W, 3
        image = cv2.Canny(image, low_thresh, high_thresh)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        prompt = [p + ", best quality, extremely detailed" for p in prompt]

        generator = [torch.Generator(device="cpu").manual_seed(seed) for i in range(len(prompt))]

        output = self.pipe(
            prompt,
            canny_image,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
            generator=generator,
            num_inference_steps=20,
        )
        
        image_gen = output['images']

        return np.array(image_gen)
    
    def re_load(self):
        self.pipe.to('cuda')
    
    def offload(self):
        self.pipe.to('cpu')


def render_random_view(mesh_path: str, output_dir: str, verbal: bool = False):
    BLENDER_PATH = 'blender'
    num_views = 1
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        # y, p = sphere_hammersley_sequence(i, num_views, offset)
        pitchs.append((np.random.rand() * 0.5) * 0.2 + 0.2)
        yaws.append(np.random.rand() * 2 * np.pi)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    _args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'hunyuan21_texture_blender.py'),
        '--',
        '--views', json.dumps(views),
        '--object', mesh_path,
        '--resolution', '512',
        '--output_folder', output_dir,
        '--engine', 'CYCLES',
    ]
    if mesh_path.endswith('.blend'):
        _args.insert(1, mesh_path)
    
    if not verbal: # default
        call(_args, stdout=DEVNULL, stderr=DEVNULL) 
    else:
        call(_args) 

    
def get_unique_obj_id(obj):
    data = pickle.dumps(obj)
    return hashlib.sha256(data).hexdigest()
    
    
def load_triangle_mesh(mesh_path: str):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    
    
def run_hunyuan_paint(mesh_path: str, output_path: str, prompt: str, 
                      control_net: Optional[ControlNet] = None,  # Pass a ControlNet object to avoid reinitialization when batch processing
                      hunyuan3d_paint_pipeline: Optional[Hunyuan3DPaintPipeline] = None,  # Pass Hunyuan3DPaintPipeline object
                      work_dir: str = 'hunyuan21_texture_cache', seed: int = 42, verbal: bool = False):
    os.makedirs(work_dir, exist_ok=True)
    verts, faces = load_triangle_mesh(mesh_path)
    uid = get_unique_obj_id({'vert': verts, 'face': faces})
    work_dir_local = os.path.join(work_dir, uid)
    os.makedirs(work_dir_local, exist_ok=True)
    print(f'Using unique id {uid} for this input. Find cache results in {work_dir_local}.')

    # Render random view
    render_random_view(mesh_path, work_dir_local, verbal=verbal)
    if not os.path.exists(os.path.join(work_dir_local, '000.png')):
        raise Exception('Failed to render random view. Check verbal message for debugging info.')
    random_view_path = os.path.join(work_dir_local, '000.png')
    
    # Run Controlnet-Canny
    if control_net is None:
        control_net = ControlNet(variant='canny')
    random_view_image = cv2.imread(random_view_path, cv2.IMREAD_UNCHANGED)
    random_view_mask = random_view_image[..., -1] < 128
    image_controlnet = control_net.run_controlnet_canny(random_view_image, [prompt], seed=seed)[0]  # h, w, 3
    image_controlnet[random_view_mask] += 255 - image_controlnet[random_view_mask]
    plt.imsave(os.path.join(work_dir_local, 'reference.png'), image_controlnet)
    print(f'ControlNet-Canny result saved to {os.path.join(work_dir_local, "reference.png")}.')
    
    # Run Hunyuan3D-2.1 (https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
    if hunyuan3d_paint_pipeline is None:    
        hunyuan3d_paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
    glb_path = hunyuan3d_paint_pipeline(mesh_path=mesh_path, image_path=os.path.join(work_dir_local, 'reference.png'), output_mesh_path=output_path, use_remesh=False).replace('.obj', '.glb')
    
    breakpoint()
    return glb_path

    
if __name__ == '__main__':
    
    mesh_path = '/afs/cs.stanford.edu/u/alexhe/projects/neurok/TRELLIS/table_seq_proc/interp/00000.obj'
    output_path = 'table.obj'
    prompt = 'An antique oak table with legs and a front drawer with a brass handle. Rustic and elegant design.'

    control_net = ControlNet(variant='canny')  # Initialize ControlNet
    hunyuan3d_paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
    run_hunyuan_paint(mesh_path, output_path, prompt, 
                      control_net=control_net, hunyuan3d_paint_pipeline=hunyuan3d_paint_pipeline, verbal=False)  # Assume mesh has z-axis as up, this would affect the randomly rendered view
