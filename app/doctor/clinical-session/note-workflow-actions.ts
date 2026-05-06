"use server";

import { Prisma } from "@prisma/client";
import { currentUser } from "@clerk/nextjs/server";
import { revalidatePath } from "next/cache";
import OpenAI from "openai";

import { prisma } from "@/lib/prisma";
import {
  buildLlmInstructionForTemplate,
  buildStrictJsonShapeExample,
  renderNotePreviewFromObject,
  validateAndNormalizeLlmPayload,
} from "../templates/template-engine";
import {
  defaultNoteNormalizationSettings,
  type NoteNormalizationSettings,
  type SoapTemplate,
  type TemplateField,
  type TemplateFieldType,
  type TemplateProfileContext,
} from "../templates/types";

const db = prisma as any;

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

import { mapRecordToSoapTemplate } from "@/lib/template-utils";

type NormalizedTranscriptSegment = {
  text: string;
  speaker: string;
  start: number;
  end: number;
  role?: string;
};

function normalizeTranscriptSegments(raw: unknown): NormalizedTranscriptSegment[] {
  if (!Array.isArray(raw)) {
    return [];
  }

  return raw
    .filter((segment) => segment && typeof segment === "object")
    .map((segment: any) => ({
      text: String(segment.text || "").trim(),
      speaker: String(segment.speaker || "Speaker 0"),
      start: typeof segment.start === "number" ? segment.start : 0,
      end: typeof segment.end === "number" ? segment.end : 0,
      role: typeof segment.role === "string" ? segment.role : undefined,
    }))
    .filter((segment) => segment.text.length > 0);
}

function buildTranscriptText(segments: NormalizedTranscriptSegment[]): string {
  return segments
    .map((segment) => {
      const label = segment.role || segment.speaker;
      return `[${label}] ${segment.text}`;
    })
    .join("\n");
}

function parseStrictShapeExample(template: SoapTemplate): Record<string, string> {
  const strictShape = buildStrictJsonShapeExample(template);
  try {
    return JSON.parse(strictShape) as Record<string, string>;
  } catch {
    return {};
  }
}

type NotePatientMetadata = {
  patient_name: string;
  patient_date_of_birth: string;
  date_of_birth: string;
  dob: string;
  patient_id: string;
  visit_date: string;
};

function formatDateForMetadata(value: Date | null | undefined): string {
  if (!value) return "";
  return value.toISOString().slice(0, 10);
}

function buildPatientMetadata(appointment: {
  date: Date;
  patient?: {
    id?: string | null;
    dateOfBirth?: Date | null;
    user?: {
      name?: string | null;
    } | null;
  } | null;
}): NotePatientMetadata {
  const patientName = appointment.patient?.user?.name?.trim() || "";
  const patientDateOfBirth = formatDateForMetadata(appointment.patient?.dateOfBirth ?? null);
  const visitDate = formatDateForMetadata(appointment.date);
  const rawPatientId = String(appointment.patient?.id ?? "").trim();
  const shortPatientId = rawPatientId ? rawPatientId.slice(-4) : "";

  return {
    patient_name: patientName,
    patient_date_of_birth: patientDateOfBirth,
    date_of_birth: patientDateOfBirth,
    dob: patientDateOfBirth,
    patient_id: shortPatientId,
    visit_date: visitDate,
  };
}

export type ActiveNoteTemplateOption = {
  id: string;
  name: string;
  description: string;
  template: SoapTemplate;
};

type ActiveNoteTemplatesResult = {
  success: boolean;
  error?: string;
  templates?: ActiveNoteTemplateOption[];
  defaultTemplateId?: string;
};

type GenerateTemplateNoteResult = {
  success: boolean;
  error?: string;
  templateId?: string;
  templateName?: string;
  noteText?: string;
  noteData?: Record<string, unknown>;
  noteMetadata?: NotePatientMetadata;
};

type SaveTemplateNoteDraftResult = {
  success: boolean;
  error?: string;
  templateId?: string;
  noteText?: string;
  noteData?: Record<string, unknown>;
  noteMetadata?: NotePatientMetadata;
};

export async function getActiveNoteTemplatesForSession(): Promise<ActiveNoteTemplatesResult> {
  const user = await currentUser();
  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  const dbUser = await prisma.user.findUnique({
    where: { clerkId: user.id },
    select: { id: true },
  });

  if (!dbUser) {
    return { success: false, error: "User not found" };
  }

  const templates = await db.noteTemplate.findMany({
    where: {
      source: "PERSONAL",
      userId: dbUser.id,
      isActive: true,
    },
    include: {
      fields: {
        orderBy: { fieldOrder: "asc" },
      },
    },
    orderBy: [{ updatedAt: "desc" }, { createdAt: "desc" }],
  });

  return {
    success: true,
    templates: templates.map((template: any) => ({
      id: template.id,
      name: template.name,
      description: template.description ?? "",
      template: mapRecordToSoapTemplate(template),
    })),
    defaultTemplateId: templates[0]?.id,
  };
}

export async function generateAppointmentNoteFromTemplate(
  appointmentId: string,
  templateId: string,
): Promise<GenerateTemplateNoteResult> {
  const user = await currentUser();
  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  const dbUser = await prisma.user.findUnique({
    where: { clerkId: user.id },
    select: { id: true },
  });

  if (!dbUser) {
    return { success: false, error: "User not found" };
  }

  const doctor = await prisma.doctorProfile.findFirst({
    where: { userId: dbUser.id },
    select: { id: true },
  });

  if (!doctor) {
    return { success: false, error: "Doctor profile not found" };
  }

  const [appointment, templateRecord] = await Promise.all([
    prisma.appointment.findFirst({
      where: {
        id: appointmentId,
        doctorId: doctor.id,
      },
      select: {
        id: true,
        date: true,
        reason: true,
        transcript: true,
        patient: {
          select: {
            id: true,
            dateOfBirth: true,
            user: {
              select: {
                name: true,
              },
            },
          },
        },
      },
    }),
    db.noteTemplate.findFirst({
      where: {
        id: templateId,
        source: "PERSONAL",
        userId: dbUser.id,
        isActive: true,
      },
      include: {
        fields: {
          orderBy: { fieldOrder: "asc" },
        },
      },
    }),
  ]);

  if (!appointment) {
    return { success: false, error: "Appointment not found" };
  }

  if (!templateRecord) {
    return { success: false, error: "Selected template is unavailable. Use an active personal template." };
  }

  const transcriptSegments = normalizeTranscriptSegments(appointment.transcript);
  const transcriptText = buildTranscriptText(transcriptSegments);

  if (!transcriptText.trim()) {
    return { success: false, error: "No transcript found. Upload or record transcription before generating a note." };
  }

  const template = mapRecordToSoapTemplate(templateRecord);
  const patientMetadata = buildPatientMetadata(appointment);

  const payload = {
    appointment_id: appointment.id,
    doctor_id: doctor.id,
    template: {
      id: template.id,
      name: template.name,
      description: template.description,
      prompt_directives: template.promptDirectives ?? "",
      header: template.header,
      footer: template.footer,
      header_text_align: template.headerTextAlign,
      normalization: template.normalization ?? defaultNoteNormalizationSettings,
      profile_context: template.profileContext,
      fields: template.bodySchema.fields.map((field) => ({
        key: field.key,
        label: field.label,
        type: field.type,
        required: field.required,
        guidance: field.guidance ?? null,
        hint: field.hint ?? null,
        fallback_policy: field.fallbackPolicy ?? "empty",
      })),
      llm_instruction: buildLlmInstructionForTemplate(template),
      strict_shape_example: parseStrictShapeExample(template),
    },
    transcript_segments: transcriptSegments,
    transcript_text: transcriptText,
    metadata: {
      appointment_reason: appointment.reason ?? "",
      appointment_date: appointment.date.toISOString(),
      patient_name: patientMetadata.patient_name,
      patient_date_of_birth: patientMetadata.patient_date_of_birth,
      patient_id: patientMetadata.patient_id,
      visit_date: patientMetadata.visit_date,
    },
  };

  try {
    const fieldKeys = payload.template.fields.map((f: any) => f.key);
    const shapeHint = payload.template.strict_shape_example || Object.fromEntries(fieldKeys.map((k: string) => [k, "string (required)"]));

    const generationPrompt =
      `Generate structured clinical note JSON from transcript.\n` +
      `Return ONLY valid JSON object (no markdown, no explanations).\n` +
      `Use exactly the required keys and no extras.\n\n` +
      `Template instruction:\n${payload.template.llm_instruction}\n\n` +
      `Required keys:\n${JSON.stringify(fieldKeys, null, 2)}\n\n` +
      `Shape hint:\n${JSON.stringify(shapeHint, null, 2)}\n\n` +
      `Metadata:\n${JSON.stringify(payload.metadata || {}, null, 2)}\n\n` +
      `Transcript:\n${payload.transcript_text}`;

    const raw = await gemmaChat([{ role: "user", content: generationPrompt }]);

    let noteData: Record<string, unknown>;
    try {
      noteData = JSON.parse(stripJsonFences(raw));
    } catch {
      throw new Error("Provider returned non-JSON note payload");
    }

    if (!noteData || typeof noteData !== "object" || Object.keys(noteData).length === 0) {
      throw new Error("Provider returned empty note payload");
    }

    const generationResult = {
      note_data: noteData,
      generated_at: Math.floor(Date.now() / 1000),
    };

    const parsed = validateAndNormalizeLlmPayload(template, generationResult.note_data ?? {});
    if (!parsed.success) {
      const firstIssue = parsed.error.issues[0];
      return {
        success: false,
        error: firstIssue?.message || "Generated note did not match template schema",
      };
    }

    const normalizedNoteData = parsed.data as Record<string, unknown>;
    const renderedNoteText = renderNotePreviewFromObject(template, normalizedNoteData, patientMetadata);

    const persistedNotePayload = {
      version: 2,
      mode: "template-generated",
      template: {
        id: template.id,
        name: template.name,
      },
      generatedAt:
        typeof generationResult.generated_at === "number"
          ? new Date(generationResult.generated_at * 1000).toISOString()
          : new Date().toISOString(),
      noteMetadata: patientMetadata,
      noteData: normalizedNoteData,
      noteText: renderedNoteText,
    } as Prisma.JsonObject;

    await prisma.appointment.update({
      where: { id: appointment.id },
      data: {
        soapNote: persistedNotePayload,
      },
    });

    revalidatePath(`/doctor/clinical-session/${appointment.id}`);
    revalidatePath("/doctor/dashboard");

    return {
      success: true,
      templateId: template.id,
      templateName: template.name,
      noteText: renderedNoteText,
      noteData: normalizedNoteData,
      noteMetadata: patientMetadata,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to generate note from template";
    console.error("generateAppointmentNoteFromTemplate failed", error);
    return {
      success: false,
      error: message,
    };
  }
}

export async function saveAppointmentTemplateNoteDraft(
  appointmentId: string,
  templateId: string,
  noteDataDraft: Record<string, unknown>,
): Promise<SaveTemplateNoteDraftResult> {
  const user = await currentUser();
  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  const dbUser = await prisma.user.findUnique({
    where: { clerkId: user.id },
    select: { id: true },
  });

  if (!dbUser) {
    return { success: false, error: "User not found" };
  }

  const doctor = await prisma.doctorProfile.findFirst({
    where: { userId: dbUser.id },
    select: { id: true },
  });

  if (!doctor) {
    return { success: false, error: "Doctor profile not found" };
  }

  const [appointment, templateRecord] = await Promise.all([
    prisma.appointment.findFirst({
      where: {
        id: appointmentId,
        doctorId: doctor.id,
      },
      select: {
        id: true,
        date: true,
        patient: {
          select: {
            id: true,
            dateOfBirth: true,
            user: {
              select: {
                name: true,
              },
            },
          },
        },
      },
    }),
    db.noteTemplate.findFirst({
      where: {
        id: templateId,
        source: "PERSONAL",
        userId: dbUser.id,
        isActive: true,
      },
      include: {
        fields: {
          orderBy: { fieldOrder: "asc" },
        },
      },
    }),
  ]);

  if (!appointment) {
    return { success: false, error: "Appointment not found" };
  }

  if (!templateRecord) {
    return { success: false, error: "Selected template is unavailable. Use an active personal template." };
  }

  const template = mapRecordToSoapTemplate(templateRecord);
  const patientMetadata = buildPatientMetadata(appointment);
  const parsed = validateAndNormalizeLlmPayload(template, noteDataDraft ?? {});

  if (!parsed.success) {
    const firstIssue = parsed.error.issues[0];
    return {
      success: false,
      error: firstIssue?.message || "Edited note does not match template schema",
    };
  }

  const normalizedNoteData = parsed.data as Record<string, unknown>;
  const renderedNoteText = renderNotePreviewFromObject(template, normalizedNoteData, patientMetadata);

  const persistedNotePayload = {
    version: 2,
    mode: "template-edited",
    template: {
      id: template.id,
      name: template.name,
    },
    generatedAt: new Date().toISOString(),
    noteMetadata: patientMetadata,
    noteData: normalizedNoteData,
    noteText: renderedNoteText,
  } as Prisma.JsonObject;

  await prisma.appointment.update({
    where: { id: appointment.id },
    data: {
      soapNote: persistedNotePayload,
    },
  });

  revalidatePath(`/doctor/clinical-session/${appointment.id}`);
  revalidatePath("/doctor/dashboard");

  return {
    success: true,
    templateId: template.id,
    noteText: renderedNoteText,
    noteData: normalizedNoteData,
    noteMetadata: patientMetadata,
  };
}
