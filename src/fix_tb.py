# save as fix_tensorboard.py
import os
import tensorboard as tb

def fix_duplicate_plugins():
    # Find the plugins directory
    tb_path = os.path.dirname(tb.__file__)
    plugin_path = os.path.join(tb_path, 'plugins')
    
    print(f"TensorBoard installed at: {tb_path}")
    print("Checking for duplicate projector plugins...")
    
    # Check for projector plugin duplicates
    count = 0
    for root, dirs, files in os.walk(plugin_path):
        for file in files:
            if 'projector' in file and 'plugin' in file:
                count += 1
                print(f"Found: {os.path.join(root, file)}")
    
    print(f"Found {count} projector plugin files.")
    print("To fix this issue, reinstall TensorBoard using: pip install --upgrade tensorboard")

if __name__ == "__main__":
    fix_duplicate_plugins()