import prisma from '../prisma';
import { Job } from '@prisma/client';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken, getGemmaApiKey, getQuantizationCacheDir } from '../paths';
import { resolvePythonPath } from '../pythonPath';
const isWindows = process.platform === 'win32';

const appendJobLog = (logPath: string, message: string) => {
  fs.appendFile(logPath, message, error => {
    if (error) console.error('Error writing to job log:', error);
  });
};

const startAndWatchJob = (job: Job, sampleOnly: boolean = false) => {
  // starts and watches the job asynchronously
  return new Promise<void>(async resolve => {
    const jobID = job.id;

    // setup the training
    const trainingRoot = await getTrainingFolder();

    const trainingFolder = path.join(trainingRoot, job.name);
    if (!fs.existsSync(trainingFolder)) {
      fs.mkdirSync(trainingFolder, { recursive: true });
    }

    // make the config file
    const configPath = path.join(trainingFolder, '.job_config.json');

    //log to path
    const logPath = path.join(trainingFolder, 'log.txt');

    try {
      // if the log path exists, move it to a folder called logs and rename it {num}_log.txt, looking for the highest num
      // if the log path does not exist, create it
      if (fs.existsSync(logPath)) {
        const logsFolder = path.join(trainingFolder, 'logs');
        if (!fs.existsSync(logsFolder)) {
          fs.mkdirSync(logsFolder, { recursive: true });
        }

        // safe move helper: try rename, then copy+unlink, then timestamped copy; never throw
        const safeMoveLog = (src: string, destDir: string) => {
          try {
            let num = 0;
            let destPath = path.join(destDir, `${num}_log.txt`);
            while (fs.existsSync(destPath)) {
              num++;
              destPath = path.join(destDir, `${num}_log.txt`);
            }
            try {
              fs.renameSync(src, destPath);
              return;
            } catch (err: any) {
              // If rename fails (common on Windows when file is locked), try copy + unlink
              console.warn('rename failed when moving log file, attempting copy+unlink', err?.code);
              try {
                fs.copyFileSync(src, destPath);
                try {
                  fs.unlinkSync(src);
                } catch (unlinkErr: any) {
                  // couldn't remove original (probably locked) — that's fine, we left a copy
                  console.warn('Could not unlink original log after copy, leaving original in place', unlinkErr?.code);
                }
                return;
              } catch (copyErr: any) {
                // If copy also fails, try a timestamped fallback copy name
                console.warn('copy failed when moving log file, attempting timestamped copy', copyErr?.code);
                try {
                  const tsName = `${Date.now()}_log.txt`;
                  const fallback = path.join(destDir, tsName);
                  fs.copyFileSync(src, fallback);
                  return;
                } catch (fallbackErr: any) {
                  // Give up — log the error and continue; we don't want to block job startup for a log file
                  console.error('Failed to move or copy log file; skipping move. Errors:', err, copyErr, fallbackErr);
                  return;
                }
              }
            }
          } catch (e) {
            console.error('Unexpected error while attempting to move log file:', e);
            return;
          }
        };

        // perform safe move
        safeMoveLog(logPath, logsFolder);
      }
    } catch (e) {
      console.error('Error moving log file:', e);
    }

    // update the config dataset path
    const jobConfig = JSON.parse(job.job_config);
    jobConfig.config.process[0].sqlite_db_path = path.join(TOOLKIT_ROOT, 'aitk_db.db');

    // write the config file
    fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

    const pythonPath = resolvePythonPath();

    const runFilePath = path.join(TOOLKIT_ROOT, 'run_ui.py');
    if (!fs.existsSync(runFilePath)) {
      console.error(`run_ui.py not found at path: ${runFilePath}`);
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: run_ui.py not found`,
        },
      });
      return;
    }

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
      PYTHONUNBUFFERED: '1', // write Python output immediately so log tail isn't lost on a crash
    };

    if (sampleOnly) {
      additionalEnv.AITK_SAMPLE_ONLY = '1';
      // Pass the job's current status so Python can restore it after sample-only completes
      additionalEnv.AITK_PREVIOUS_STATUS = job.status;
    }

    // HF_TOKEN
    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      additionalEnv.HF_TOKEN = hfToken;
    }

    // GEMMA_API_KEY — injected so Python can use it without embedding it in the config YAML
    const gemmaApiKey = await getGemmaApiKey();
    if (gemmaApiKey && gemmaApiKey.trim() !== '') {
      additionalEnv.GEMMA_API_KEY = gemmaApiKey;
    }

    // AITK_QUANTIZATION_CACHE_DIR — only injected when the job opts in via cache_quantized_model
    if (jobConfig?.config?.process?.[0]?.model?.cache_quantized_model) {
      const quantCacheDir = await getQuantizationCacheDir();
      additionalEnv.AITK_QUANTIZATION_CACHE_DIR = quantCacheDir;
    }

    // Add the --log argument to the command
    const args = [runFilePath, configPath, '--log', logPath];

    try {
      let subprocess;

      if (isWindows) {
        // Spawn Python directly on Windows so the process can survive parent exit
        subprocess = spawn(pythonPath, args, {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
          detached: true,
          windowsHide: true,
          stdio: 'ignore', // don't tie stdio to parent; run_ui.py writes its own --log file
        });
      } else {
        // For non-Windows platforms, fully detach and ignore stdio so it survives daemon-like
        subprocess = spawn(pythonPath, args, {
          detached: true,
          stdio: 'ignore', // don't tie stdio to parent; run_ui.py writes its own --log file
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
        });
      }

      // Handle failures where the child process could not be started.
      subprocess.once('error', error => {
        const message = `Error launching job process: ${error.message}`;
        console.error(message);
        appendJobLog(logPath, `${message}\n`);
        void prisma.job
          .update({
            where: { id: jobID },
            data: { status: 'error', info: message, pid: null },
          })
          .catch(updateError => {
            console.error('Error updating job after process launch failure:', updateError);
          });
      });

      // Record abnormal termination and repair jobs Python could not update itself.
      subprocess.once('exit', (code, signal) => {
        if (code === 0) return;

        const result = signal ? `signal ${signal}` : `exit code ${code}`;
        const message = `Job process terminated with ${result}.`;
        appendJobLog(logPath, `\n${message}\n`);
        void prisma.job
          .updateMany({
            where: { id: jobID, status: 'running' },
            data: { status: 'error', info: message, pid: null },
          })
          .catch(updateError => {
            console.error('Error updating job after abnormal process exit:', updateError);
          });
      });

      // Save the PID to the database and a file for future management (stop/inspect)
      const pid = subprocess.pid ?? null;
      if (pid != null) {
        await prisma.job.update({
          where: { id: jobID },
          data: { pid },
        });
      }
      try {
        fs.writeFileSync(path.join(trainingFolder, 'pid.txt'), String(pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }

      // Important: let the child run independently of this Node process.
      if (subprocess.unref) {
        subprocess.unref();
      }

      // The child remains independent; these listeners only record failures
      // while the worker is alive.
    } catch (error: any) {
      // Handle any exceptions during process launch
      console.error('Error launching process:', error);
      appendJobLog(logPath, `Error launching job process: ${error?.message || 'Unknown error'}\n`);

      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: ${error?.message || 'Unknown error'}`,
        },
      });
      return;
    }
    // Resolve the promise immediately after starting the process
    resolve();
  });
};

export default async function startJob(jobID: string, sampleOnly: boolean = false) {
  const job: Job | null = await prisma.job.findUnique({
    where: { id: jobID },
  });
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }
  // update job status to 'running', this will run sync so we don't start multiple jobs.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'running',
      stop: false,
      return_to_queue: false,
      info: sampleOnly ? 'Generating samples...' : 'Starting job...',
    },
  });
  // start and watch the job asynchronously so the cron can continue
  startAndWatchJob(job, sampleOnly);
}
