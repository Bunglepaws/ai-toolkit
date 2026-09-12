import { startMonitor } from '@/server/monitor';
import { MonitorSample } from '@/types';

// SSE stream of system stats. On connect: an `init` event with the 2-minute
// rolling history plus the latest full sample; then a `sample` event every
// MONITOR_TICK_MS. Auth is the normal middleware bearer check, so the client
// uses fetch (EventSource can't send headers).
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Vast/nginx-style proxies buffer until ~4KiB or until the connection closes.
// X-Accel-Buffering: no is often ignored. A comment pad after each event
// forces a flush so the dashboard is not stuck on the empty init snapshot.
const SSE_PAD = `:${' '.repeat(4096)}\n\n`;

export async function GET(request: Request) {
  const monitor = startMonitor();
  const encoder = new TextEncoder();
  let unsubscribe: (() => void) | null = null;
  let heartbeat: ReturnType<typeof setInterval> | null = null;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const enqueue = (text: string) => {
        controller.enqueue(encoder.encode(text));
      };
      const send = (event: string, data: unknown) => {
        enqueue(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
        enqueue(SSE_PAD);
      };
      const cleanup = () => {
        if (heartbeat) {
          clearInterval(heartbeat);
          heartbeat = null;
        }
        unsubscribe?.();
        unsubscribe = null;
        try {
          controller.close();
        } catch {
          // already closed
        }
      };
      try {
        send('init', monitor.getInit());
      } catch {
        cleanup();
        return;
      }
      heartbeat = setInterval(() => {
        try {
          enqueue(`: keepalive\n\n`);
          enqueue(SSE_PAD);
        } catch {
          cleanup();
        }
      }, 2000);
      unsubscribe = monitor.subscribe((sample: MonitorSample, serialized: string) => {
        // desiredSize goes ever more negative when the consumer stopped
        // reading. If the abort for a vanished client never propagates (dev
        // HMR reloads, proxy quirks), this is the backstop that keeps dead
        // subscribers from accumulating and eating CPU forever.
        if (controller.desiredSize !== null && controller.desiredSize < -120) {
          cleanup();
          return;
        }
        try {
          enqueue(`event: sample\ndata: ${serialized}\n\n`);
          enqueue(SSE_PAD);
        } catch {
          // Client is gone but abort hasn't fired yet
          cleanup();
        }
      });
      request.signal.addEventListener('abort', cleanup);
    },
    cancel() {
      if (heartbeat) {
        clearInterval(heartbeat);
        heartbeat = null;
      }
      unsubscribe?.();
      unsubscribe = null;
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
      'Content-Encoding': 'identity',
    },
  });
}
