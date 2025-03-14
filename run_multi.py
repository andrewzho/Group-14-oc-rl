import os
import sys
import argparse
import subprocess
import platform
import time
import signal
import random

def ensure_python_path():
    """Add project root to Python path"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)

def find_and_kill_unity_processes():
    """Find and kill any Unity processes to clean up before starting"""
    print("Cleaning up any stale Unity processes...")
    try:
        if platform.system() == "Windows":
            # Use tasklist and taskkill on Windows
            # Find Unity processes
            process = subprocess.Popen(
                "tasklist /FI \"IMAGENAME eq obstacletower*\" /FO CSV /NH", 
                shell=True, 
                stdout=subprocess.PIPE
            )
            stdout, _ = process.communicate()
            
            for line in stdout.decode().splitlines():
                if "obstacletower" in line.lower() or "unity" in line.lower():
                    try:
                        # Extract PID and kill
                        parts = line.split(',')
                        if len(parts) >= 2:
                            pid = parts[1].strip('"')
                            subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                            print(f"Killed stale Unity process with PID {pid}")
                    except:
                        pass
        else:
            # Use ps and kill on Unix-like systems
            process = subprocess.Popen(
                "ps aux | grep -i 'obstacle\\|unity' | grep -v grep", 
                shell=True, 
                stdout=subprocess.PIPE
            )
            stdout, _ = process.communicate()
            
            for line in stdout.decode().splitlines():
                try:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"Killed stale Unity process with PID {pid}")
                except:
                    pass
        
        # Wait for processes to terminate
        time.sleep(3)
        print("Cleanup complete")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def find_obstacle_tower_executable():
    """Find the Obstacle Tower executable in the expected locations"""
    potential_paths = [
        os.path.join("ObstacleTower", "obstacletower.exe"),
        os.path.join("ObstacleTower", "ObstacleTower.exe"),
        os.path.join("ObstacleTower", "ObstacleTower-v3.exe"),
        os.path.join("ObstacleTower", "ObstacleTower.x86_64"),
        os.path.join("ObstacleTower", "ObstacleTower-v3.x86_64")
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    """Parse arguments and run training script"""
    parser = argparse.ArgumentParser(description="Run multi-agent Obstacle Tower training")
    parser.add_argument("--env_path", type=str, default=None,
                       help="Path to Obstacle Tower executable")
    parser.add_argument("--num_envs", type=int, default=2,
                       help="Number of environments to run in parallel")
    parser.add_argument("--num_steps", type=int, default=100000,
                       help="Total number of steps to train for")
    parser.add_argument("--use_lstm", action="store_true",
                       help="Use LSTM-based recurrent policy")
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                       help="Device to run training on (cuda/cpu)")
    parser.add_argument("--test_only", action="store_true",
                       help="Only run tests, don't start training")
    parser.add_argument("--worker_id_base", type=int, default=None,
                       help="Base worker ID for environments (random if not specified)")
    
    args = parser.parse_args()
    
    # Find executable if not provided
    if args.env_path is None:
        args.env_path = find_obstacle_tower_executable()
        if args.env_path is None:
            print("Error: Could not find Obstacle Tower executable. Please specify with --env_path")
            return 1
    
    # Clean up any stale processes before starting
    find_and_kill_unity_processes()
    
    # Generate a random worker ID base if not specified
    if args.worker_id_base is None:
        args.worker_id_base = random.randint(1000, 9000)
        print(f"Using random worker ID base: {args.worker_id_base}")
    
    # Ensure path is properly quoted for Windows
    if platform.system() == "Windows" and " " in args.env_path:
        args.env_path = f'"{args.env_path}"'
    
    # Set up command
    if args.test_only:
        cmd = [
            sys.executable, "-m", "multi_agent.test_multi",
            "--env_path", args.env_path,
            "--num_envs", str(min(args.num_envs, 2)),  # Use at most 2 envs for testing
            "--device", args.device
        ]
    else:
        cmd = [
            sys.executable, "-m", "multi_agent.train_multi",
            "--env_path", args.env_path,
            "--num_envs", str(args.num_envs),
            "--num_steps", str(args.num_steps),
            "--device", args.device,
            "--save_interval", "10",
            "--worker_id_base", str(args.worker_id_base)
        ]
        
        if args.use_lstm:
            cmd.append("--use_lstm")
    
    # Run command
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    
    process = None
    try:
        # For Windows, use shell=True
        if platform.system() == "Windows":
            process = subprocess.Popen(cmd_str, shell=True)
        else:
            process = subprocess.Popen(cmd)
        
        process.wait()
        return_code = process.returncode
        
        # Clean up after the process finishes
        find_and_kill_unity_processes()
        
        return return_code
    except KeyboardInterrupt:
        print("Interrupted by user")
        if process:
            process.terminate()
        # Make sure to clean up after interruption
        find_and_kill_unity_processes()
        return 1
    except Exception as e:
        print(f"Error running command: {e}")
        # Clean up after error
        find_and_kill_unity_processes()
        return 1

if __name__ == "__main__":
    ensure_python_path()
    sys.exit(main()) 