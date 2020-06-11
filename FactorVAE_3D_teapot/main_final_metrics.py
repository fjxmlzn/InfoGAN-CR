if __name__ == "__main__":
    from factorvae_task_final_metrics import FactorVAETask
    from config_final_metrics import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(config=config, gpu_task_class=FactorVAETask)
    scheduler.start()