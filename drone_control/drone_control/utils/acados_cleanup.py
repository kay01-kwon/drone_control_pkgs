import os
import shutil

def cleanup_acados_files():
    """Remove generated Acados files"""
    curr_dir = os.getcwd()
    target_list = ['c_generated_code', 'acados_ocp.json']

    for target in target_list:
        path = os.path.join(curr_dir, target)
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f'[Cleanup] Removed: {path}')
            except Exception as e:
                print(f'[Cleanup] Failed to remove {path}: {e}')