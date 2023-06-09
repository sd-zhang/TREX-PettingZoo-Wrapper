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

def add_envid_to_launchlist(trex_launch_lists, env_ids=None, force_separate_ports=False):
    for trex_launch_list_nbr in range(len(trex_launch_lists)):  # for every launch list


        for client_nbr in range(len(trex_launch_lists[trex_launch_list_nbr])):

            # get the index of the client's port number
            client_args = trex_launch_lists[trex_launch_list_nbr]
            client_args = client_args[client_nbr]
            client_args = client_args[1]


            # FixMe: Steven tells me that this should not be necessary
            if force_separate_ports:
                port_is_in_client = [True for i, s in enumerate(client_args) if s.startswith('--port=')]
                if port_is_in_client:
                    port_index = [i for i, s in enumerate(client_args) if s.startswith('--port=')]
                    assert len(port_index) == 1
                    port_index = port_index[0]
                    # #make sure the port index of each launch list is unique, but also the same for each client in the launch list
                    trex_launch_lists[trex_launch_list_nbr][client_nbr][1][port_index] = '--port=' + str(
                        42069 + env_ids[trex_launch_list_nbr] * 10)

            # add the env_id to the gym agent config
            is_trader = [True for i, s in enumerate(client_args) if s.startswith('--trader=')]
            if is_trader:
                # if we haev a gym agent, find the trader info
                trader_index = [i for i, s in enumerate(client_args) if s.startswith('--trader=')]
                assert len(trader_index) == 1
                trader_index = trader_index[0]
                trader_info_old = trex_launch_lists[trex_launch_list_nbr][client_nbr][1][trader_index]
                # print('trader info:', trader_info)
                # add the env_id to the trader info

                addendum = ', '+ '"env_id": ' + str(env_ids[trex_launch_list_nbr])

                trader_info = trader_info_old[:-1] + addendum + trader_info_old[-1:]
                # print('trader info with env added:', trader_info)
                trex_launch_lists[trex_launch_list_nbr][client_nbr][1][trader_index] = trader_info

    return trex_launch_lists