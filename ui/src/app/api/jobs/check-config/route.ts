import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { exec } from 'child_process';
import { promisify } from 'util';
import OpenAI from 'openai';
import sqlite3 from 'sqlite3';
import { getTrainingFolder, getCheckConfigApiBaseUrl, getCheckConfigApiKey, getCheckConfigModel, getCheckConfigEnableWebSearch } from '@/server/settings';
import { buildCheckConfigSystemPrompt } from '@/lib/checkConfigPrompt';
import {
  getModelFamily,
  collectSampleMedia,
  collectDatasetImages,
  type MediaItem,
} from '@/lib/checkConfigMedia';
import { modelArchs } from '@/app/jobs/new/options';

const execAsync = promisify(exec);

async function collectSystemStats(): Promise<Record<string, unknown>> {
  const stats: Record<string, unknown> = {
    system_ram: {
      total_mb: Math.round(os.totalmem() / 1024 / 1024),
      free_mb: Math.round(os.freemem() / 1024 / 1024),
      used_mb: Math.round((os.totalmem() - os.freemem()) / 1024 / 1024),
    },
  };

  try {
    const { stdout } = await execAsync(
      'nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits',
      { timeout: 8000 },
    );
    stats.gpus = stdout.trim().split('\n').map(line => {
      const [index, name, memTotal, memUsed, memFree, gpuUtil, memUtil, powerDraw, powerLimit, temp] =
        line.split(', ').map(s => s.trim());
      return {
        index: parseInt(index),
        name,
        memory_total_mb: parseInt(memTotal),
        memory_used_mb: parseInt(memUsed),
        memory_free_mb: parseInt(memFree),
        gpu_utilization_pct: parseInt(gpuUtil),
        memory_utilization_pct: parseInt(memUtil),
        power_draw_w: parseFloat(powerDraw),
        power_limit_w: parseFloat(powerLimit),
        temperature_c: parseInt(temp),
      };
    });
  } catch {
    // nvidia-smi unavailable — omit GPU stats
  }

  return stats;
}

export const runtime = 'nodejs';

function buildWebSearchQuery(processType: string, jobConfig: any): string {
  const arch = modelArchs.find(a => a.name === processType);
  const modelLabel = arch?.label ?? processType;

  const process0 = jobConfig?.config?.process?.[0] ?? {};
  const optimizer: string = process0?.train?.optimizer ?? process0?.network?.optimizer ?? '';
  const nonStandardOptimizers = ['automagic3', 'prodigy', 'dadaptation', 'sophia', 'adan', 'came'];
  const optimizerSuffix = nonStandardOptimizers.includes(optimizer) ? ` ${optimizer} optimizer` : '';

  return `${modelLabel} LoRA fine-tuning training ai-toolkit${optimizerSuffix}`;
}

function getOllamaHost(baseURL: string): string | null {
  if (!baseURL) return null;
  if (baseURL.includes('anthropic.com') || baseURL.includes('openai.com')) return null;
  return baseURL.replace(/\/v1\/?$/, '').replace(/\/$/, '');
}

function formatWebSearchResults(results: Array<{ title?: string; url?: string; content?: string }>): string | null {
  if (!results.length) return null;
  const lines = [
    'Web search results (supporting context — prefer these over internal knowledge for current best practices):',
  ];
  for (const [i, r] of results.entries()) {
    const title = r.title || 'Untitled';
    const url = r.url || '';
    const snippet = (r.content || '').replace(/\s+/g, ' ').slice(0, 400);
    lines.push(
      `${i + 1}. ${title}${url ? `\n   URL: ${url}` : ''}${snippet ? `\n   ${snippet}` : ''}`,
    );
  }
  return lines.join('\n');
}

async function fetchOllamaWebSearch(
  ollamaHost: string,
  apiKey: string,
  query: string,
): Promise<string | null> {
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (apiKey && apiKey !== 'no-key') headers['Authorization'] = `Bearer ${apiKey}`;
    const res = await fetch(`${ollamaHost}/api/web_search`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ query, max_results: 5 }),
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) {
      if (res.status === 401) {
        console.warn('[check-config] Ollama web search returned 401 — set API Key in Settings');
      } else if (res.status === 429) {
        console.warn('[check-config] Ollama web search rate-limited (429) — proceeding without web context');
      }
      return null;
    }
    const data = await res.json();
    return formatWebSearchResults(data.results ?? []);
  } catch {
    return null;
  }
}

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 10_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close((err) => (err ? reject(err) : resolve()));
  });
}

function sseEvent(data: Record<string, unknown>): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

export async function POST(request: NextRequest) {
  const baseURL = await getCheckConfigApiBaseUrl();
  if (!baseURL) {
    return NextResponse.json(
      { error: 'Check Config API URL is not configured. Add it in Settings.' },
      { status: 503 },
    );
  }

  let body: { jobId?: string; includeImages?: boolean };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { jobId, includeImages = false } = body;
  if (!jobId) {
    return NextResponse.json({ error: 'jobId is required' }, { status: 400 });
  }

  const job = await prisma.job.findUnique({ where: { id: jobId } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const stream = new ReadableStream({
    async start(controller) {
      const emit = (data: Record<string, unknown>) => {
        controller.enqueue(new TextEncoder().encode(sseEvent(data)));
      };

      try {
        // Stage 1: system stats
        emit({ type: 'progress', message: 'Collecting system stats…' });
        const systemStats = await collectSystemStats();

        // Stage 2: loss curve
        emit({ type: 'progress', message: 'Reading loss curve…' });
        let lossPoints: { step: number; value: number }[] = [];
        if (job.step > 0) {
          try {
            const trainingFolder = await getTrainingFolder();
            const logPath = path.join(trainingFolder, job.name, 'loss_log.db');
            if (fs.existsSync(logPath)) {
              const db = openDb(logPath);
              try {
                const rows = await all<{ step: number; value: number | null }>(
                  db,
                  `SELECT m.step, m.value_real AS value
                   FROM metrics m
                   WHERE m.key = 'loss'
                   ORDER BY m.step DESC
                   LIMIT 200`,
                );
                lossPoints = rows
                  .filter((r) => r.value != null)
                  .map((r) => ({ step: r.step, value: r.value! }))
                  .reverse();
              } finally {
                await closeDb(db);
              }
            }
          } catch {
            // Loss data is optional
          }
        }

        // Stage 3: visual media
        const jobConfig = JSON.parse(job.job_config);
        let datasetStats: Record<string, unknown> = {};
        try {
          datasetStats = JSON.parse((job as any).dataset_stats || '{}');
        } catch { /* ignore */ }

        const processType: string = jobConfig?.config?.process?.[0]?.type ?? '';
        const modelFamily = getModelFamily(processType);

        let sampleMedia: MediaItem[] = [];
        let datasetMedia: MediaItem[] = [];
        let ffmpegUnavailable = false;

        if (includeImages) {
          emit({ type: 'progress', message: 'Collecting sample images…' });
          try {
            const trainingFolder = await getTrainingFolder();
            const sampleResult = await collectSampleMedia(trainingFolder, job.name, modelFamily);
            sampleMedia = sampleResult.items;
            ffmpegUnavailable = sampleResult.ffmpegUnavailable ?? false;
          } catch { /* best-effort */ }

          emit({ type: 'progress', message: 'Collecting dataset images…' });
          try {
            const datasetPaths: string[] = (jobConfig?.config?.process?.[0]?.datasets ?? [])
              .map((d: any) => d?.folder_path)
              .filter(Boolean);
            datasetMedia = await collectDatasetImages(datasetPaths, modelFamily);
          } catch { /* best-effort */ }
        }

        const hasImages = sampleMedia.length > 0 || datasetMedia.length > 0;

        const textContext = JSON.stringify(
          {
            model_family: modelFamily,
            job_config: jobConfig,
            dataset_stats: Object.keys(datasetStats).length > 0 ? datasetStats : null,
            current_step: job.step,
            total_steps: job.total_steps,
            loss_curve_last_200_steps: lossPoints.length > 0 ? lossPoints : null,
            system_stats: systemStats,
            visual_analysis_requested: includeImages,
            visual_analysis_available: hasImages,
            ...(ffmpegUnavailable && { note: 'ffmpeg unavailable — video frame extraction skipped; config-only analysis for video model' }),
            ...(!hasImages && includeImages && modelFamily !== 'audio' && { note: 'No sample images found in the samples folder — config-only analysis' }),
          },
          null,
          2,
        );

        type ContentPart =
          | { type: 'text'; text: string }
          | { type: 'image_url'; image_url: { url: string }; label?: string };

        const userContent: ContentPart[] = [{ type: 'text', text: textContext }];

        if (sampleMedia.length > 0) {
          userContent.push({ type: 'text', text: 'Recent sample outputs from this training run:' });
          for (const img of sampleMedia) {
            userContent.push({ type: 'image_url', image_url: { url: img.dataUrl }, label: img.label });
          }
        }

        if (datasetMedia.length > 0) {
          userContent.push({ type: 'text', text: 'Sample images from the training dataset:' });
          for (const img of datasetMedia) {
            userContent.push({ type: 'image_url', image_url: { url: img.dataUrl }, label: img.label });
          }
        }

        const apiKey = (await getCheckConfigApiKey()) || 'no-key';
        const model = (await getCheckConfigModel()) || 'claude-sonnet-5';
        const enableWebSearch = await getCheckConfigEnableWebSearch();

        // Stage 4: web search
        let finalTextContext = textContext;
        const ollamaHost = getOllamaHost(baseURL);
        if (enableWebSearch && ollamaHost) {
          emit({ type: 'progress', message: 'Running web search…' });
          const searchQuery = buildWebSearchQuery(processType, jobConfig);
          const webContext = await fetchOllamaWebSearch(ollamaHost, apiKey, searchQuery);
          if (webContext) {
            finalTextContext = `${webContext}\n\n---\n\nJob configuration and training context:\n${textContext}`;
            if (userContent.length > 0 && userContent[0].type === 'text') {
              userContent[0] = { type: 'text', text: finalTextContext };
            }
          }
        }

        // Stage 5: AI call
        emit({ type: 'progress', message: 'Sending to AI model…' });
        const client = new OpenAI({ baseURL, apiKey });

        // Ollama-specific overrides passed as extra body fields (ignored by other providers).
        // - think: false  → skip chain-of-thought reasoning (Qwen3 etc.)
        // - options.num_ctx: 8192 → override the server's global context length (default 262144
        //   allocates ~36 GB of KV cache, leaving almost no VRAM for model weights)
        const extraParams = ollamaHost ? { think: false, options: { num_ctx: 8192 } } : {};

        let raw = '';
        try {
          const response = await client.chat.completions.create({
            model,
            messages: [
              { role: 'system', content: buildCheckConfigSystemPrompt(modelFamily, hasImages) },
              { role: 'user', content: hasImages ? userContent : finalTextContext },
            ],
            max_tokens: 4096,
            ...extraParams,
          } as any);
          raw = response.choices[0]?.message?.content || '';
        } catch (err: any) {
          emit({ type: 'error', message: `LLM API call failed: ${err?.message || String(err)}` });
          controller.close();
          return;
        }

        // Stage 6: parse response
        emit({ type: 'progress', message: 'Parsing response…' });
        const stripped = raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();

        let findings: unknown[];
        try {
          const parsed = JSON.parse(stripped);
          findings = Array.isArray(parsed) ? parsed : (parsed?.findings ?? []);
        } catch {
          emit({ type: 'error', message: 'Failed to parse AI response', raw } as any);
          controller.close();
          return;
        }

        emit({ type: 'done', findings });
      } catch (err: any) {
        emit({ type: 'error', message: err?.message || String(err) });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

export async function GET() {
  const baseURL = await getCheckConfigApiBaseUrl();
  return NextResponse.json({ enabled: !!baseURL });
}
