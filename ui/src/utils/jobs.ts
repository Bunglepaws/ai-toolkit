import { JobConfig } from '@/types';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

export const startJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/start`)
      .then(res => res.data)
      .then(data => {
        console.log('Job started:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error starting job:', error);
        reject(error);
      });
  });
};

export const stopJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/stop`)
      .then(res => res.data)
      .then(data => {
        console.log('Job stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping job:', error);
        reject(error);
      });
  });
};

export const saveAndRequeueJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/save_and_requeue`)
      .then(res => res.data)
      .then(data => {
        console.log('Job save-and-requeue requested:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error requesting save-and-requeue:', error);
        reject(error);
      });
  });
};

export const gracefulStopJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/stop_graceful`)
      .then(res => res.data)
      .then(data => {
        console.log('Job graceful stop requested:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error requesting graceful stop:', error);
        reject(error);
      });
  });
};

export const saveJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/save`)
      .then(res => res.data)
      .then(data => {
        console.log('Job save requested:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error requesting job save:', error);
        reject(error);
      });
  });
};

export const saveAndPauseJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/save_and_pause`)
      .then(res => res.data)
      .then(data => {
        console.log('Job save-and-pause requested:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error requesting job save-and-pause:', error);
        reject(error);
      });
  });
};

export const stopSampleJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/stop_sample`)
      .then(res => res.data)
      .then(data => {
        console.log('Sample abort requested:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error requesting sample abort:', error);
        reject(error);
      });
  });
};

export const sampleJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/sample`)
      .then(res => res.data)
      .then(data => {
        console.log('Job sample requested:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error requesting job sample:', error);
        reject(error);
      });
  });
};

export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/delete`)
      .then(res => res.data)
      .then(data => {
        console.log('Job deleted:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error deleting job:', error);
        reject(error);
      });
  });
};

export const saveJobNow = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/save_now`)
      .then(res => res.data)
      .then(data => {
        console.log('Job set to save on next step:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error setting job to save on next step:', error);
        reject(error);
      });
  });
};

export const sampleJobNow = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/sample_now`)
      .then(res => res.data)
      .then(data => {
        console.log('Job set to sample on next step:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error setting job to sample on next step:', error);
        reject(error);
      });
  });
};

export const markJobAsStopped = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/mark_stopped`)
      .then(res => res.data)
      .then(data => {
        console.log('Job marked as stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error marking job as stopped:', error);
        reject(error);
      });
  });
};

export const reorderJobToIndex = (jobID: string, targetIndex: number) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/reorder`, { targetIndex })
      .then(res => res.data)
      .then(data => {
        console.log(`Job ${jobID} reordered to index ${targetIndex}:`, data);
        resolve();
      })
      .catch(error => {
        console.error(`Error reordering job ${jobID} to index ${targetIndex}:`, error);
        reject(error);
      });
  });
};

export const reorderJob = (jobID: string, direction: 'up' | 'down') => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/reorder`, { direction })
      .then(res => res.data)
      .then(data => {
        console.log(`Job ${jobID} reordered ${direction}:`, data);
        resolve();
      })
      .catch(error => {
        console.error(`Error reordering job ${jobID}:`, error);
        reject(error);
      });
  });
};

export const clearJobAlerts = (jobID: string): Promise<void> => {
  return apiClient.get(`/api/jobs/${jobID}/clear_alerts`).then(() => undefined);
};

export const listCheckpoints = (jobID: string) => {
  return apiClient.get(`/api/jobs/${jobID}/checkpoints`);
};

export const resumeFromCheckpoint = (jobID: string, step: number, deleteSamples = false) => {
  return apiClient.post(`/api/jobs/${jobID}/resume-from`, { step, deleteSamples });
};

export const getJobConfig = (job: Job) => {
  return JSON.parse(job.job_config) as JobConfig;
};

export const getAvaliableJobActions = (job: Job, isAnyJobRunning: boolean = false, hasSamples: boolean = false) => {
  const jobConfig = getJobConfig(job);
  const isStopping = job.stop && job.status === 'running';
  const isSaving = job.save_now && job.status === 'running';
  const isSampling = job.sample && job.status === 'running';

  // Busy if it's currently saving or sampling. Stopping is its own state.
  const isBusy = isSaving || isSampling;

  const canDelete = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  let canEdit = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canRemoveFromQueue = job.status === 'queued';

  // Stop should ALWAYS be available if the job is running and not already stopping.
  // We want to be able to kill a job even if it's stuck saving or sampling.
  const canStop = job.status === 'running' && !isStopping;

  // Cannot save if already busy or stopping
  const canSave = job.status === 'running' && !isBusy && !isStopping;

  // edit sample prompts at any time except when actively generating samples
  const canEditSample = !isSampling;

  let canStart = ['stopped', 'error'].includes(job.status) && !isStopping;
  // can resume if more steps were added
  if (job.status === 'completed' && (jobConfig.config.process[0].train?.steps || 0) > job.step && !isStopping) {
    canStart = true;
  }
  if (!['train', 'caption'].includes(job.job_type)) {
    canEdit = false;
  }
  // True when Python is actively generating images (info string set by UITrainer.sample())
  const isActivelySampling = job.status === 'running' && (job.info?.includes('Generating images') ?? false);

  // Only allow triggering samples when in the actual training loop — not during model loading,
  // quantization, dataset encoding, or other startup phases.
  const isInTrainingLoop = !job.info || job.info === '' || job.info.startsWith('Training');

  const canSampleRunning = job.status === 'running' && !isBusy && !isStopping && isInTrainingLoop;
  const canSampleStopped = !['running', 'queued'].includes(job.status) && !isAnyJobRunning && hasSamples && !job.sample;
  const canSample = canSampleRunning || canSampleStopped;

  return { canDelete, canEdit, canEditSample, canStop, canStart, canRemoveFromQueue, canSave, canSample, isBusy, isStopping, isActivelySampling };
};

export const getNumberOfSamples = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].sample?.prompts?.length || 0;
};

export const getTotalSteps = (job: Job) => {
  if (job.total_steps != null) {
    return job.total_steps;
  }
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].train?.steps || 0;
};
