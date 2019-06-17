if __name__ == "__main__":
    from gan_task_generate_latent_trans import GANTask
    from config_generate_latent_trans import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(config=config, gpu_task_class=GANTask)
    scheduler.start()