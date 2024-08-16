# Run all the steps with a single command
# Usage: python run_all.py

import os
import sys
import subprocess

# Run the steps
def run_command(command):
    print("Running command: " + command)
    subprocess.run(command, shell=True)
    
def main():
    run_command("python preprocessing.py")
    run_command("python dataloader.py")
    run_command("python model.py")
    run_command("python train.py")
    run_command("python eval.py")
    
if __name__ == "__main__":
    main()
    
# Run the script
# python3 run_all.py