from typing import Union, OrderedDict

from toolkit.config import get_config


def get_job(
        config_path: Union[str, dict, OrderedDict],
        name=None,
        sample_only=False
):
    config = get_config(config_path, name)
    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job_type = config['job']
    job = None
    if job_type == 'extract':
        from jobs import ExtractJob
        job = ExtractJob(config)
    elif job_type == 'train':
        from jobs import TrainJob
        job = TrainJob(config)
    elif job_type == 'mod':
        from jobs import ModJob
        job = ModJob(config)
    elif job_type == 'generate':
        from jobs import GenerateJob
        job = GenerateJob(config)
    elif job_type == 'extension':
        from jobs import ExtensionJob
        job = ExtensionJob(config)
    else:
        raise ValueError(f'Unknown job type {job_type}')

    if job is not None:
        job.sample_only = sample_only

    return job


def run_job(
        config: Union[str, dict, OrderedDict],
        name=None
):
    job = get_job(config, name)
    job.run()
    job.cleanup()
