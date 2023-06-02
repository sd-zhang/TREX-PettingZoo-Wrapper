import os
import sys
import subprocess
import TREX_Core

# ToDo: move this to some utils
def prep_trex(config_name):
    #trex_path = os.path.join(os.getcwd(), 'TREX_Core') #ToDo: make sure this is the right path!
    trex_path = TREX_Core.__path__[0]
    #runner expects to be running in the TREX_Core directory, so we swap.
    cur_dir = os.getcwd()
    os.chdir(trex_path)
    runner = TREX_Core._utils.runner.Runner(config_name, resume=False, purge=False, path=trex_path)
    os.chdir(cur_dir)

    config = runner.modify_config(simulation_type='training')
    launch_list = runner.make_launch_list(config)
    return launch_list

# ToDo: move this to some utils
def run_subprocess(args : list):
    subprocess.run([sys.executable, args[0], *args[1]])
