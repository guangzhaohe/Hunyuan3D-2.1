from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")
    
    
def run_hunyuan3d(mesh_path: str, image_path: str, max_num_view: int = 6, resolution: int = 512):
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    output_mesh_path = paint_pipeline(mesh_path=mesh_path, image_path=image_path)    


if __name__ == "__main__":

    max_num_view = 6  # can be 6 to 9
    resolution = 512  # can be 768 or 512

    run_hunyuan3d(mesh_path='/afs/cs.stanford.edu/u/alexhe/projects/neurok/TRELLIS/table_seq_proc/interp/00000.obj', 
                  image_path='/afs/cs.stanford.edu/u/alexhe/projects/neurok/TRELLIS/data/Trellis_Cache/texture_gen/4489f2e9b4b579660e3f028b85cba2cabcc1b7a7/control_net.png',
                  max_num_view=max_num_view, resolution=resolution)
