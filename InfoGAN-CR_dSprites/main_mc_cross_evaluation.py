if __name__ == "__main__":
    from gan_task_mc_cross_evaluation import GANTask
    from config_mc import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(config=config, gpu_task_class=GANTask)
    scheduler.start()