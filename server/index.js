import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { bearerAuth } from "hono/bearer-auth";
import http from "node:http";
import { Buffer } from "node:buffer";

// --- Config ---
const PORT = parseInt(process.env.PORT || "3000");
const API_KEY = process.env.API_KEY || "dev-secret-key";
const SILENCE_TIMEOUT_MS = parseInt(process.env.SILENCE_TIMEOUT_MS || "60000");
const PYTHON_SOCKET = process.env.PYTHON_SOCKET || "/tmp/pipeline.sock";
const PYTHON_HTTP =
  process.env.PYTHON_HTTP || "http://localhost:8000"; // fallback for local dev
const CHECK_INTERVAL_MS = parseInt(process.env.CHECK_INTERVAL_MS || "10000");

// --- Session state per user ---
/** @type {Map<string, { buffer: Buffer[], lastVoiceAt: number, startAt: number, isActive: boolean }>} */
const sessions = new Map();

// --- Hono app ---
const app = new Hono();

// Health check
app.get("/health", (c) => c.json({ status: "ok" }));

// Audio endpoint with Bearer auth
app.post("/audio", bearerAuth({ token: API_KEY }), async (c) => {
  const uid = c.req.query("uid") || "default";
  const sampleRate = parseInt(c.req.query("sample_rate") || "16000");

  const body = await c.req.arrayBuffer();
  const chunk = Buffer.from(body);

  if (chunk.length === 0) {
    return c.json({ status: "empty" }, 200);
  }

  // Ask Python VAD if there's speech in this chunk
  const hasSpeech = await vadCheck(chunk, sampleRate);

  const now = Date.now();
  let session = sessions.get(uid);

  if (hasSpeech && !session) {
    // New session
    session = {
      buffer: [chunk],
      lastVoiceAt: now,
      startAt: now,
      isActive: true,
    };
    sessions.set(uid, session);
    console.log(`[${uid}] New session started`);
  } else if (hasSpeech && session) {
    // Append to existing session
    session.buffer.push(chunk);
    session.lastVoiceAt = now;
  } else if (!hasSpeech && session) {
    // Silence during active session - still append (captures trailing audio)
    session.buffer.push(chunk);
  }
  // !hasSpeech && !session → ignore

  return c.json({ status: "ok", has_speech: hasSpeech, session_active: !!session });
});

// --- Silence timer: check all sessions periodically ---
setInterval(() => {
  const now = Date.now();
  for (const [uid, session] of sessions.entries()) {
    if (!session.isActive) continue;
    const silenceMs = now - session.lastVoiceAt;
    if (silenceMs >= SILENCE_TIMEOUT_MS) {
      console.log(
        `[${uid}] Session finalized (${silenceMs / 1000}s silence, ${session.buffer.length} chunks)`
      );
      session.isActive = false;
      finalizeSession(uid, session);
    }
  }
}, CHECK_INTERVAL_MS);

// --- Finalize: send full buffer to Python pipeline ---
async function finalizeSession(uid, session) {
  const fullBuffer = Buffer.concat(session.buffer);
  const durationSec = fullBuffer.length / (16000 * 2); // PCM16 = 2 bytes/sample
  console.log(
    `[${uid}] Processing ${durationSec.toFixed(1)}s of audio (${(fullBuffer.length / 1024).toFixed(0)} KB)`
  );

  try {
    const result = await callPython("/process", fullBuffer, {
      uid,
      start_at: new Date(session.startAt).toISOString(),
    });
    console.log(`[${uid}] Pipeline result:`, result);
  } catch (err) {
    console.error(`[${uid}] Pipeline error:`, err.message);
  } finally {
    sessions.delete(uid);
  }
}

// --- Python API communication ---
async function vadCheck(audioBuffer, sampleRate) {
  try {
    const result = await callPython("/vad-check", audioBuffer, {
      sample_rate: sampleRate,
    });
    return result.has_speech === true;
  } catch (err) {
    console.error("VAD check failed:", err.message);
    // On failure, assume speech to avoid losing audio
    return true;
  }
}

async function callPython(path, audioBuffer, metadata = {}) {
  const url = new URL(path, PYTHON_HTTP);
  const queryParams = new URLSearchParams(
    Object.entries(metadata).map(([k, v]) => [k, String(v)])
  );
  url.search = queryParams.toString();

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: audioBuffer,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Python API ${path} returned ${response.status}: ${text}`);
  }

  return response.json();
}

// --- Start server ---
serve({ fetch: app.fetch, port: PORT }, (info) => {
  console.log(`Webhook server listening on port ${info.port}`);
  console.log(`Silence timeout: ${SILENCE_TIMEOUT_MS / 1000}s`);
  console.log(`Python API: ${PYTHON_HTTP}`);
});
