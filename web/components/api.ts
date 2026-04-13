const API_BASE = process.env.NEXT_PUBLIC_OMNI_API_BASE ?? "http://localhost:8000";

function getCookie(name: string): string {
  if (typeof document === "undefined") return "";
  const escaped = name.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&");
  const m = document.cookie.match(new RegExp(`(?:^|; )${escaped}=([^;]*)`));
  return m ? decodeURIComponent(m[1]) : "";
}

export type FileRecord = {
  original_name: string;
  stored_name: string;
  type: "image" | "pdf";
  size: number;
  page_count?: number;
};

export type SessionSummary = {
  session_id: string;
  created_at: string;
  title: string;
};

export type HeuristicHint = {
  domain: string;
  confidence: string;
};

export type SessionDetail = {
  meta: Record<string, unknown>;
  files: FileRecord[];
  draft_spec: string;
  questions_md: string;
  answers_md: string;
  spec_md: string;
  react_code: string;
  heuristic?: HeuristicHint;
};

export type StartResult = {
  session_id: string;
  files: FileRecord[];
  draft_spec: string;
  questions_md: string;
  heuristic?: HeuristicHint;
};

async function apiFetch(path: string, init?: RequestInit) {
  const method = (init?.method ?? "GET").toUpperCase();
  const headers = new Headers(init?.headers ?? {});
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
    const csrf = getCookie("omni_csrf");
    if (csrf) headers.set("X-CSRF-Token", csrf);
  }
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    credentials: "include",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

export async function login(password: string) {
  return apiFetch("/auth/login", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ password }),
  });
}

export async function listSessions(): Promise<SessionSummary[]> {
  return apiFetch("/sessions");
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  return apiFetch(`/sessions/${sessionId}`);
}

export async function startSession(
  files: File[],
  userPrompt: string,
): Promise<StartResult> {
  const form = new FormData();
  form.append("user_prompt", userPrompt);
  for (const f of files) {
    form.append("files", f);
  }
  return apiFetch("/sessions/start", { method: "POST", body: form });
}

export async function addFiles(
  sessionId: string,
  files: File[],
): Promise<{ ok: boolean; new_files: FileRecord[]; total_files: FileRecord[] }> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  return apiFetch(`/sessions/${sessionId}/files`, { method: "POST", body: form });
}

export async function refineSpec(
  sessionId: string,
  answersMd: string,
): Promise<{ spec_md: string }> {
  return apiFetch("/sessions/refine", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, answers_md: answersMd }),
  });
}

export async function shipToCloud(
  sessionId: string,
  modelName: string,
  instructions: string,
): Promise<{ react_code: string; output_type?: string }> {
  return apiFetch("/sessions/ship", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, model_name: modelName, instructions }),
  });
}

export async function exportSessionZip(sessionId: string) {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/export`, {
    method: "GET",
    credentials: "include",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Export failed: ${res.status}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `omni-session-${sessionId}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
