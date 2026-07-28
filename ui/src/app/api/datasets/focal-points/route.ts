import { NextRequest } from 'next/server';
import fs from 'fs';
import { getCheckConfigApiBaseUrl, getCheckConfigApiKey, getCheckConfigModel } from '@/server/settings';
import OpenAI from 'openai';

export const runtime = 'nodejs';

const VALID_ANCHORS = [
  'top-left', 'top-center', 'top-right',
  'center-left', 'center', 'center-right',
  'bottom-left', 'bottom-center', 'bottom-right',
];

function normalizeAnchor(raw: string): string {
  const lower = raw.toLowerCase().trim().replace(/\s+/g, '-');
  if (VALID_ANCHORS.includes(lower)) return lower;
  // Search for any valid anchor substring in the response
  for (const anchor of VALID_ANCHORS) {
    if (lower.includes(anchor)) return anchor;
  }
  return 'center';
}

async function imageToDataUrl(filePath: string): Promise<string | null> {
  try {
    const data = await fs.promises.readFile(filePath);
    const isPng = data[0] === 0x89 && data[1] === 0x50;
    const mimeType = isPng ? 'image/png' : 'image/jpeg';
    return `data:${mimeType};base64,${data.toString('base64')}`;
  } catch {
    return null;
  }
}

export async function POST(request: NextRequest) {
  let body: { imagePaths?: string[] };
  try {
    body = await request.json();
  } catch {
    return new Response('Invalid JSON', { status: 400 });
  }

  const imagePaths = body.imagePaths ?? [];
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
      };

      const baseURL = await getCheckConfigApiBaseUrl();
      if (!baseURL) {
        for (const p of imagePaths) {
          send({ path: p, anchor: 'center', source: 'fallback' });
        }
        send({ done: true });
        controller.close();
        return;
      }

      const apiKey = (await getCheckConfigApiKey()) || 'no-key';
      const model = (await getCheckConfigModel()) || 'claude-sonnet-5';
      const client = new OpenAI({ baseURL, apiKey });

      for (const imagePath of imagePaths) {
        try {
          const dataUrl = await imageToDataUrl(imagePath);
          if (!dataUrl) {
            send({ path: imagePath, anchor: 'center', source: 'fallback' });
            continue;
          }

          const response = await client.chat.completions.create({
            model,
            messages: [
              {
                role: 'system',
                content:
                  'You detect the position of the main subject in images. Reply with ONLY one of these exact words: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right. No other text.',
              },
              {
                role: 'user',
                content: [
                  { type: 'image_url', image_url: { url: dataUrl } },
                  { type: 'text', text: 'Where is the main subject?' },
                ] as any,
              },
            ],
            max_tokens: 20,
          });

          const raw = response.choices[0]?.message?.content ?? '';
          const anchor = normalizeAnchor(raw);
          send({ path: imagePath, anchor, source: 'ai' });
        } catch {
          send({ path: imagePath, anchor: 'center', source: 'fallback' });
        }
      }

      send({ done: true });
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
}
