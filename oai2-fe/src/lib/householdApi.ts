/** Typed client for household identity REST (`/api/v1/reid/...`). */

export type ResidentRecord = {
  uuid: string;
  stable_id: number;
  display_name: string;
  created_ts: number;
  embedding_count?: number;
  gallery_embeddings?: number;
};

export type ResidentListResponse = {
  residents: ResidentRecord[];
  count: number;
};

export type ResidentEnrollResponse = {
  resident: ResidentRecord;
  applied: boolean;
};

export type IdentityHealth = {
  household_mode: boolean;
  resident_count: number;
  visitor_count: number;
  provisional_count: number;
  mint_visitor_count: number;
  promote_resident_count: number;
  false_share_blocked_count: number;
  overlap_permit_grant_count: number;
  overlap_permit_deny_count: number;
  gallery_ids: number;
  gallery_quality_reject_count: number;
  mnn_reject_count: number;
  active_unique: number;
  residents: ResidentRecord[];
  metrics?: Record<string, unknown>;
};

export type SuggestCandidate = {
  a: number;
  b: number;
  sim: number;
  pose_sim?: number | null;
  canonical: number;
  preferred_canonical: number;
  a_embedding_count: number;
  b_embedding_count: number;
  blocked: boolean;
  block_reason?: string | null;
};

export type SuggestResponse = {
  candidates: SuggestCandidate[];
  default_min_sim: number;
};

export type MergeResponse = {
  applied: boolean;
  src: number;
  dst: number;
  canonical: number;
  reason?: string | null;
  aliases: Record<number, number>;
};

export class HouseholdApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail || `HTTP ${status}`);
    this.name = 'HouseholdApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function parseError(resp: Response): Promise<HouseholdApiError> {
  let detail = `HTTP ${resp.status}`;
  try {
    const body = await resp.json();
    if (typeof body?.detail === 'string') detail = body.detail;
    else if (Array.isArray(body?.detail)) detail = body.detail.map((d: any) => d?.msg || String(d)).join('; ');
    else if (body?.detail != null) detail = JSON.stringify(body.detail);
  } catch {
    try {
      const text = await resp.text();
      if (text) detail = text.slice(0, 240);
    } catch {
      // keep default
    }
  }
  return new HouseholdApiError(resp.status, detail);
}

function joinUrl(base: string, path: string): string {
  const root = (base || '').replace(/\/$/, '');
  return `${root}${path}`;
}

async function getJson<T>(base: string, path: string): Promise<T> {
  const resp = await fetch(joinUrl(base, path), { method: 'GET' });
  if (!resp.ok) throw await parseError(resp);
  return (await resp.json()) as T;
}

async function sendJson<T>(base: string, path: string, method: string, body?: unknown): Promise<T> {
  const resp = await fetch(joinUrl(base, path), {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!resp.ok) throw await parseError(resp);
  return (await resp.json()) as T;
}

export function listResidents(base: string): Promise<ResidentListResponse> {
  return getJson(base, '/api/v1/reid/residents');
}

export function enrollResident(
  base: string,
  body: { display_name: string; stable_id?: number; visitor_id?: number },
): Promise<ResidentEnrollResponse> {
  return sendJson(base, '/api/v1/reid/residents/enroll', 'POST', body);
}

export function patchResident(
  base: string,
  uuid: string,
  body: { display_name: string },
): Promise<ResidentEnrollResponse> {
  return sendJson(base, `/api/v1/reid/residents/${encodeURIComponent(uuid)}`, 'PATCH', body);
}

export function deleteResident(base: string, uuid: string): Promise<ResidentEnrollResponse> {
  return sendJson(base, `/api/v1/reid/residents/${encodeURIComponent(uuid)}`, 'DELETE');
}

export function getIdentityHealth(base: string): Promise<IdentityHealth> {
  return getJson(base, '/api/v1/reid/identity_health');
}

export function suggestAliases(
  base: string,
  body: { min_sim?: number; limit?: number; require_inactive?: boolean } = {},
): Promise<SuggestResponse> {
  return sendJson(base, '/api/v1/reid/aliases/suggest', 'POST', {
    limit: 20,
    require_inactive: true,
    ...body,
  });
}

export function mergeAliases(
  base: string,
  body: { a: number; b: number; canonical?: number; force?: boolean },
): Promise<MergeResponse> {
  return sendJson(base, '/api/v1/reid/aliases/merge', 'POST', {
    append_embeddings: true,
    force: false,
    ...body,
  });
}

/** Visitor public IDs live in 1000–1031; provisional typically 9000+. */
export function classifyStableId(sid: number): 'resident' | 'visitor' | 'provisional' | 'other' {
  if (sid >= 1 && sid < 1000) return 'resident';
  if (sid >= 1000 && sid <= 1031) return 'visitor';
  if (sid >= 9000) return 'provisional';
  return 'other';
}
