import { autonomyInputSchema, decideAutonomyWithJev, autonomyJevConfiguration, AutonomyJevError } from "../../../lib/autonomy-jev";

export const runtime = "nodejs";
export const maxDuration = 12;
export const dynamic = "force-dynamic";

const MAX_BODY_BYTES = 8 * 1024;
const RATE_WINDOW_MS = 60_000;
const MAX_REQUESTS_PER_WINDOW = 20;
const MAX_IP_BUCKETS = 2048;

// Best effort within one process. Serverless instances do not share this map;
// this limits basic accidental abuse, not authenticated or billed user quotas.
const requestsByIp = new Map<string, { count: number; expiresAt: number }>();

function consumeRequest(request: Request) {
  const now = Date.now();
  const ip = (request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "local").slice(0, 128);
  for (const [key, bucket] of requestsByIp) {
    if (bucket.expiresAt <= now) requestsByIp.delete(key);
  }
  const bucket = requestsByIp.get(ip);
  if (bucket) {
    if (bucket.count >= MAX_REQUESTS_PER_WINDOW) return false;
    bucket.count += 1;
    return true;
  }
  if (requestsByIp.size >= MAX_IP_BUCKETS) return false;
  requestsByIp.set(ip, { count: 1, expiresAt: now + RATE_WINDOW_MS });
  return true;
}

function json(body: unknown, status = 200, headers: Record<string, string> = {}) {
  return Response.json(body, { status, headers: { "Cache-Control": "no-store", ...headers } });
}

function hasSameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  if (!origin) return true; // Non-browser clients need not send Origin.
  try {
    const parsedOrigin = new URL(origin);
    if (parsedOrigin.origin !== origin || !/^https?:$/.test(parsedOrigin.protocol)) return false;

    // Next can construct request.url with its internal hostname (for example,
    // localhost when the browser visits 127.0.0.1). Host is the browser-facing
    // authority, including Vercel preview and custom domains. Never let a
    // caller-supplied x-forwarded-host replace it.
    const requestUrl = new URL(request.url);
    const host = request.headers.get("host") || requestUrl.host;
    if (!host || /[\/\\@\s,#?]/.test(host)) return false;

    // Vercel terminates TLS before invoking the function. Outside Vercel,
    // forwarded headers are untrusted and cannot change the allowed scheme.
    const forwardedProtocol = process.env.VERCEL === "1" ? request.headers.get("x-forwarded-proto") : null;
    if (forwardedProtocol && forwardedProtocol !== "http" && forwardedProtocol !== "https") return false;
    const protocol = forwardedProtocol ? `${forwardedProtocol}:` : requestUrl.protocol;
    return origin === new URL(`${protocol}//${host}`).origin;
  } catch {
    return false;
  }
}

async function readBody(request: Request): Promise<unknown> {
  if (Number(request.headers.get("content-length") || "0") > MAX_BODY_BYTES) {
    throw new AutonomyJevError(413, "That observation is too large. Send only the bounded recent memory.");
  }
  if (!request.body) throw new AutonomyJevError(400, "Send the current simulator observation and memory.");

  const reader = request.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let size = 0;
  let body = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY_BYTES) {
        await reader.cancel();
        throw new AutonomyJevError(413, "That observation is too large. Send only the bounded recent memory.");
      }
      body += decoder.decode(value, { stream: true });
    }
    body += decoder.decode();
    return JSON.parse(body);
  } catch (error) {
    if (error instanceof AutonomyJevError) throw error;
    throw new AutonomyJevError(400, "The request is not valid JSON.");
  } finally {
    reader.releaseLock();
  }
}

export async function GET() {
  return json(autonomyJevConfiguration());
}

export async function POST(request: Request) {
  try {
    if (!hasSameOrigin(request)) {
      return json({ error: "Send autonomy requests from the simulator's own page." }, 403);
    }
    if (!/^application\/json(?:\s*;|$)/i.test(request.headers.get("content-type") || "")) {
      return json({ error: "Use application/json for autonomy requests." }, 415);
    }
    if (!consumeRequest(request)) {
      return json({ error: "Too many autonomy decisions. Wait a moment, then try again." }, 429, { "Retry-After": "60" });
    }

    const input = autonomyInputSchema.safeParse(await readBody(request));
    if (!input.success) {
      return json({ error: "Send a valid bounded simulator observation and memory." }, 400);
    }
    return json(await decideAutonomyWithJev(input.data, { signal: request.signal }));
  } catch (error) {
    if (error instanceof AutonomyJevError) return json({ error: error.message }, error.status);
    return json({ error: "The simulator could not process that observation. Autonomy is stopped." }, 500);
  }
}
