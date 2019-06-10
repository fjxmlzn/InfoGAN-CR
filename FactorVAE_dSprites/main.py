if __name__ == "__main__":
    from factorvae_task import FactorVAETask
    from config import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(config=config, gpu_task_class=FactorVAETask)
    scheduler.start()