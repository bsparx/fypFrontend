"use server";

import { Prisma } from "@prisma/client";
import { currentUser } from "@clerk/nextjs/server";
import { revalidatePath } from "next/cache";
import OpenAI from "openai";
import { prisma } from "@/lib/prisma";

const VOXTRAL_BASE = "https://api.mistral.ai/v1/audio/transcriptions";
const VOXTRAL_MODEL = "voxtral-mini-latest";

const GEMMA_BASE_URL =
  process.env.GEMMA_BASE_URL?.trim() ||
  "https://muddasirjaved666--example-gemma-4-e2b-autoround-it-infer-780f02.modal.run/v1";
const GEMMA_API_KEY = "sk-dummy-anything";
const GEMMA_MODEL =
  process.env.GEMMA_MODEL?.trim() || "cyankiwi/gemma-4-E4B-it-AWQ-INT4";

const openai = new OpenAI({
  apiKey: GEMMA_API_KEY,
  baseURL: GEMMA_BASE_URL,
});

async function gemmaChat(
  messages: Array<{ role: string; content: string }>
): Promise<string> {
  const res = await openai.chat.completions.create({
    model: GEMMA_MODEL,
    messages: messages as any,
    temperature: 0.1,
    response_format: { type: "json_object" },
  });
  return res.choices[0]?.message?.content || "{}";
}

function stripJsonFences(text: string): string {
  return text
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/```\s*$/, "")
    .trim();
}

function normaliseSpeaker(raw: unknown): string {
  if (raw == null) return "Speaker 0";
  if (typeof raw === "number") return `Speaker ${raw}`;
  const s = String(raw).trim();
  const m = s.match(/\d+/);
  if (m) return `Speaker ${parseInt(m[0], 10)}`;
  return s || "Speaker 0";
}

function inferFilenameAndContentType(
  recordingUrl: string,
  contentTypeHeader: string | null
): [string, string] {
  const url = new URL(recordingUrl);
  const pathName = decodeURIComponent(url.pathname || "");
  const fileName = pathName.split("/").pop() || "recording.webm";

  const headerType = (contentTypeHeader || "").split(";")[0].trim().toLowerCase();
  if (headerType) return [fileName, headerType];

  const ext = fileName.slice(fileName.lastIndexOf(".")).toLowerCase();
  const map: Record<string, string> = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
  };
  return [fileName, map[ext] || "application/octet-stream"];
}

async function voxtralTranscribe(
  audioBytes: ArrayBuffer,
  diarize: boolean,
  filename: string,
  contentType: string,
  strict: boolean = false,
  maxRetries: number = 2
): Promise<{
  text: string;
  segments: Array<{ text: string; speaker: string; start: number; end: number }>;
}> {
  const apiKey = process.env.MISTRAL_API_KEY?.trim() || "";

  const dataPayload: Record<string, string> = {
    model: VOXTRAL_MODEL,
    response_format: "verbose_json",
  };

  if (diarize) {
    dataPayload.diarize = "true";
    dataPayload.timestamp_granularities = "segment";
  } else {
    dataPayload.language = "en";
  }

  const blob = new Blob([audioBytes], { type: contentType });
  const formData = new FormData();
  formData.append("file", blob, filename);
  for (const [k, v] of Object.entries(dataPayload)) {
    formData.append(k, v);
  }

  const transientStatuses = new Set([408, 429, 500, 502, 503, 504, 520, 522, 524]);

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const resp = await fetch(VOXTRAL_BASE, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
        body: formData,
      });

      if (resp.status !== 200) {
        const bodyPreview = await resp.text();
        const rayId = resp.headers.get("cf-ray");
        console.error(
          `[Voxtral API HTTP ${resp.status} Error] attempt=${attempt + 1}/${maxRetries + 1} ray=${rayId}: ${bodyPreview.slice(0, 2000)}`
        );

        if (transientStatuses.has(resp.status) && attempt < maxRetries) {
          await new Promise((r) => setTimeout(r, 1500 * 2 ** attempt));
          continue;
        }

        if (strict) {
          throw new Error(`Voxtral upstream error ${resp.status}: ${bodyPreview.slice(0, 2000)}`);
        }
        return { text: "", segments: [] };
      }

      const result = (await resp.json()) as any;
      const text = result.text || "";
      const rawSegments = result.segments || [];
      const segments: Array<{ text: string; speaker: string; start: number; end: number }> = [];

      let missingSpeakerCount = 0;
      for (const s of rawSegments) {
        const spkRaw = s.speaker;
        const segText = s.text || "";
        const start = typeof s.start === "number" ? s.start : parseFloat(s.start) || 0;
        const end = typeof s.end === "number" ? s.end : parseFloat(s.end) || 0;

        if (diarize && spkRaw == null) missingSpeakerCount++;

        segments.push({
          text: String(segText).trim(),
          speaker: normaliseSpeaker(spkRaw),
          start,
          end,
        });
      }

      if (diarize && segments.length > 0 && missingSpeakerCount === segments.length) {
        return await llmDiarizeFallback(segments, text);
      }

      return { text, segments };
    } catch (err: any) {
      if (err.message?.startsWith("Voxtral upstream error")) throw err;
      console.error(
        `[Voxtral Connection Trace Error] attempt=${attempt + 1}/${maxRetries + 1}: ${err}`
      );
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 1500 * 2 ** attempt));
        continue;
      }
      if (strict) throw err;
      return { text: "", segments: [] };
    }
  }

  if (strict) throw new Error("Voxtral transcription failed after retries");
  return { text: "", segments: [] };
}

async function llmDiarizeFallback(
  segments: Array<{ text: string; speaker: string; start: number; end: number }>,
  fullText: string
): Promise<{
  text: string;
  segments: Array<{ text: string; speaker: string; start: number; end: number }>;
}> {
  console.warn("Mistral API missed tags. Initiating LLM Dictionary Fallback!");
  const scriptBlock = segments.map((s, i) => `[${i}]: ${s.text}`).join("\n");
  const prompt =
    `You are a medical transcript separator.\n` +
    `The following transcript segments lost their speaker labels. Read the conversational flow ` +
    `and intuitively assign each segment to 'Doctor' or 'Patient'.\n\n` +
    `NOTE: A single speaker might span multiple consecutive segments if they paused. ` +
    `Do NOT blindly alternate if the flow suggests one person continued speaking.\n\n` +
    `TRANSCRIPT SEGMENTS:\n${scriptBlock}\n\n` +
    `Return ONLY a valid JSON DICTIONARY mapping the segment integer string to its role. ` +
    `Example: {"0": "Doctor", "1": "Patient", "2": "Patient"}. NO MARKDOWN.`;

  const raw = await gemmaChat([{ role: "user", content: prompt }]);
  try {
    const roleMapping = JSON.parse(stripJsonFences(raw));
    if (roleMapping && typeof roleMapping === "object") {
      let lastKnownRole = "Doctor";
      for (let i = 0; i < segments.length; i++) {
        const role = roleMapping[String(i)];
        if (role === "Doctor" || role === "Patient" || role === "Nurse") {
          lastKnownRole = role;
        }
        segments[i].speaker = lastKnownRole;
      }
      console.info("Fallback Net Successful. Hand-mapped dictionary applied.");
      return { text: fullText, segments };
    }
  } catch (e) {
    console.error("Fallback Net Error parsing Dict:", e);
  }

  for (const seg of segments) {
    seg.speaker = "Speaker 0";
  }
  return { text: fullText, segments };
}

async function classifyRoles(
  segments: Array<{ text: string; speaker: string; start: number; end: number }>
): Promise<Record<string, string>> {
  const samples: Record<string, string[]> = {};
  const freshCounts: Record<string, number> = {};

  for (const seg of segments) {
    const spk = seg.speaker || "Speaker 0";
    const text = seg.text.trim();
    if (spk === "Doctor" || spk === "Patient" || spk === "Nurse") {
      continue;
    }
    freshCounts[spk] = (freshCounts[spk] || 0) + 1;
    if (!samples[spk]) samples[spk] = [];
    if (samples[spk].length < 8 && text) {
      samples[spk].push(text);
    }
  }

  const eligible = Object.entries(samples).filter(([spk]) => (freshCounts[spk] || 0) >= 2);

  const roles: Record<string, string> = {};

  for (const seg of segments) {
    if (seg.speaker === "Doctor" || seg.speaker === "Patient" || seg.speaker === "Nurse") {
      roles[seg.speaker] = seg.speaker;
    }
  }

  if (eligible.length === 0) {
    return roles;
  }

  const lines = eligible.map(([spk, txts]) => {
    const excerpt = txts.slice(0, 5).join(" | ");
    return `${spk} (${freshCounts[spk]} segments): "${excerpt}"`;
  });
  const speakerBlock = lines.join("\n");

  const prompt =
    `You are analyzing a medical consultation transcript.\n` +
    `Classify each Speaker as exactly one of: Doctor, Patient, Nurse, Guardian, Unknown.\n\n` +
    `Assign DIFFERENT roles to DIFFERENT speakers.\n\n` +
    `Speakers:\n${speakerBlock}\n\n` +
    `Reply ONLY with a valid JSON object, e.g.:\n` +
    `{"Speaker 0": "Doctor", "Speaker 1": "Patient"}\n`;

  const raw = await gemmaChat([{ role: "user", content: prompt }]);
  try {
    const parsed = JSON.parse(stripJsonFences(raw));
    if (parsed && typeof parsed === "object") {
      Object.assign(roles, parsed);
    }
  } catch {
    // ignore
  }

  return roles;
}

type PersistTranscriptResult = {
  success: boolean;
  error?: string;
  aiStatus?: string | null;
  transcriptSegments?: number;
};

function normalizeSegments(raw: unknown): Prisma.JsonArray {
  if (!Array.isArray(raw)) {
    return [];
  }

  const normalized = raw
    .filter((segment) => segment && typeof segment === "object")
    .map((segment: any) => ({
      text: String(segment.text || "").trim(),
      speaker: String(segment.speaker || "Speaker 0"),
      start: typeof segment.start === "number" ? segment.start : 0,
      end: typeof segment.end === "number" ? segment.end : 0,
      role: typeof segment.role === "string" ? segment.role : null,
    }))
    .filter((segment) => segment.text.length > 0);

  return normalized as Prisma.JsonArray;
}

export async function confirmAndSaveAppointmentTranscription(appointmentId: string): Promise<PersistTranscriptResult> {
  const user = await currentUser();
  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  const doctor = await prisma.doctorProfile.findFirst({
    where: {
      user: {
        clerkId: user.id,
      },
    },
    select: { id: true },
  });

  if (!doctor) {
    return { success: false, error: "Doctor profile not found" };
  }

  const appointment = await prisma.appointment.findFirst({
    where: {
      id: appointmentId,
      doctorId: doctor.id,
    },
    select: {
      id: true,
      recordingUrl: true,
      aiStatus: true,
    },
  });

  if (!appointment) {
    return { success: false, error: "Appointment not found" };
  }

  if (!appointment.recordingUrl) {
    return { success: false, error: "No recording found for this appointment" };
  }

  await prisma.appointment.update({
    where: { id: appointment.id },
    data: {
      aiStatus: "PROCESSING",
    },
  });

  try {
    // Download audio
    const audioResponse = await fetch(appointment.recordingUrl, { method: "GET" });
    if (!audioResponse.ok) {
      throw new Error(`Could not fetch audio: ${audioResponse.status}`);
    }
    const audioBuffer = await audioResponse.arrayBuffer();

    const contentTypeHeader = audioResponse.headers.get("content-type");
    const [filename, contentType] = inferFilenameAndContentType(appointment.recordingUrl, contentTypeHeader);

    // Transcribe
    const result = await voxtralTranscribe(audioBuffer, true, filename, contentType, true, 2);
    const { text, segments } = result;

    if (!text && segments.length === 0) {
      throw new Error("Transcription provider returned an empty payload after retries");
    }

    if (segments.length === 0 && text) {
      segments.push({ speaker: "Speaker 0", text, start: 0, end: 0 });
    }

    // Classify speaker roles
    const speakerRoles = await classifyRoles(segments);

    const annotatedSegments = segments.map((seg) => ({
      ...seg,
      role: speakerRoles[seg.speaker] || "Unknown",
    }));

    const normalizedSegments = normalizeSegments(annotatedSegments);

    await prisma.appointment.update({
      where: { id: appointment.id },
      data: {
        transcript: normalizedSegments,
        aiStatus: "COMPLETED",
      },
    });

    revalidatePath(`/doctor/clinical-session/${appointment.id}`);
    revalidatePath("/doctor/dashboard");

    return {
      success: true,
      aiStatus: "COMPLETED",
      transcriptSegments: normalizedSegments.length,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to transcribe recording";
    console.error("confirmAndSaveAppointmentTranscription failed", error);

    await prisma.appointment.update({
      where: { id: appointment.id },
      data: {
        aiStatus: "FAILED",
      },
    });

    return {
      success: false,
      error: message,
      aiStatus: "FAILED",
    };
  }
}

export async function getAppointmentTranscriptionStatus(appointmentId: string) {
  const user = await currentUser();
  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  const appointment = await prisma.appointment.findFirst({
    where: {
      id: appointmentId,
      doctor: {
        is: {
          user: {
            clerkId: user.id,
          },
        },
      },
    },
    select: {
      aiStatus: true,
      transcript: true,
      updatedAt: true,
    },
  });

  if (!appointment) {
    return { success: false, error: "Appointment not found" };
  }

  return {
    success: true,
    aiStatus: appointment.aiStatus,
    hasTranscript: Array.isArray(appointment.transcript) && appointment.transcript.length > 0,
    updatedAt: appointment.updatedAt,
  };
}
