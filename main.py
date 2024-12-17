if __name__ == '__main__':
    from TREX_Core.runner import Runner
    # configuration to be used must be under _configs
    # file name must be config name. Case sensitive.
    # it is sometimes possible to resume a simulation in case it crashes in the middle
    # however due to the complexity this is experimental and cannot be relied upon
    # runner = Runner(config='atco_1h_hc4_aggregate', resume=False, purge=True)
    # config = 'atco_1h_hc4_aggregate'
    config = 'atco_1h_hc4_perm1_bess'
    runner = Runner(config=config, resume=False, purge=True)

    # list of simulations to be performed.
    # in general, it is recommended to perform at base line and training at minimum
    simulations = {
        'baseline'
        # 'training'
        # 'validation'
    }
    # launch_list = runner.make_launch_list()
    # print(launch_list)
    # runner.run([launch_list[3]])
    runner.run_simulations(simulations)