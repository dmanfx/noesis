import React, { useEffect, useId, useState } from 'react';
import { colorForTrack, colorIdForPerson, detectCameraKey, type CameraKey } from '../lib/camera';
import {
  HouseholdApiError,
  classifyStableId,
  deleteResident,
  enrollResident,
  getIdentityHealth,
  listResidents,
  mergeAliases,
  patchResident,
  suggestAliases,
  type IdentityHealth,
  type ResidentRecord,
  type SuggestCandidate,
} from '../lib/householdApi';
import '../styles/household-identity.css';

export type HouseholdLiveTrack = {
  stable_id: number;
  tracker_id?: number;
  camera_id: string;
  zone?: string;
  display_name?: string | null;
  resident_uuid?: string | null;
  identity_kind?: string | null;
  identity_state?: string | null;
  reid_confidence?: number | null;
  embedding_present?: boolean | null;
  overlap_permit?: boolean | null;
};

type Props = {
  open: boolean;
  onClose: () => void;
  restBaseUrl: string;
  liveTracks: HouseholdLiveTrack[];
};

type Banner = { kind: 'ok' | 'warn' | 'error' | 'info'; text: string } | null;

type EnrollTarget = {
  stable_id: number;
  camera_id: string;
  kind: string;
  display_name?: string | null;
};

const kindOf = (t: HouseholdLiveTrack): string => {
  const raw = String(t.identity_kind || t.identity_state || '').toLowerCase();
  if (raw === 'resident' || raw === 'visitor' || raw === 'provisional' || raw === 'handoff') return raw;
  return classifyStableId(Number(t.stable_id));
};

const canEnrollTrack = (t: HouseholdLiveTrack): { ok: boolean; reason?: string } => {
  const kind = kindOf(t);
  if (kind === 'resident') return { ok: false, reason: 'Already a resident — rename below' };
  if (kind === 'provisional') return { ok: false, reason: 'Wait until identity confirms (visitor)' };
  if (kind === 'handoff') return { ok: false, reason: 'Handoff in progress' };
  if (kind !== 'visitor' && classifyStableId(Number(t.stable_id)) !== 'visitor') {
    return { ok: false, reason: 'Select a visitor ID (1000–1031)' };
  }
  return { ok: true };
};

const formatTs = (ts: number): string => {
  if (!ts || !Number.isFinite(ts)) return '—';
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return '—';
  }
};

const confText = (v: number | null | undefined): string => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  return v.toFixed(2);
};

export const HouseholdIdentityDrawer: React.FC<Props> = ({
  open,
  onClose,
  restBaseUrl,
  liveTracks,
}) => {
  const titleId = useId();
  const [banner, setBanner] = useState<Banner>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [householdOff, setHouseholdOff] = useState(false);
  const [residents, setResidents] = useState<ResidentRecord[]>([]);
  const [health, setHealth] = useState<IdentityHealth | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestCandidate[]>([]);
  const [enrollTarget, setEnrollTarget] = useState<EnrollTarget | null>(null);
  const [enrollName, setEnrollName] = useState('');
  const [renameUuid, setRenameUuid] = useState<string | null>(null);
  const [renameName, setRenameName] = useState('');
  const [confirmDeleteUuid, setConfirmDeleteUuid] = useState<string | null>(null);

  const refresh = async () => {
    if (!open) return;
    setLoading(true);
    setBanner(null);
    setHouseholdOff(false);
    try {
      const [resList, healthPayload] = await Promise.all([
        listResidents(restBaseUrl),
        getIdentityHealth(restBaseUrl),
      ]);
      setResidents(resList.residents || []);
      setHealth(healthPayload);
      if (!healthPayload.household_mode) {
        setHouseholdOff(true);
        setBanner({
          kind: 'warn',
          text: 'Household mode is off on the runtime. Enable NOESIS_HOUSEHOLD_IDENTITY=1 and restart.',
        });
      }
      try {
        const sug = await suggestAliases(restBaseUrl, { limit: 12, require_inactive: true });
        setSuggestions(sug.candidates || []);
      } catch {
        setSuggestions([]);
      }
    } catch (err) {
      const apiErr = err instanceof HouseholdApiError ? err : null;
      if (apiErr?.status === 400 && /household/i.test(apiErr.detail)) {
        setHouseholdOff(true);
        setBanner({
          kind: 'warn',
          text: 'Household mode is not enabled on this runtime.',
        });
      } else if (apiErr?.status === 503) {
        setBanner({
          kind: 'error',
          text: 'ReID manager unavailable — is DS8 running?',
        });
      } else {
        setBanner({
          kind: 'error',
          text: apiErr?.detail || (err instanceof Error ? err.message : 'Failed to load identity data'),
        });
      }
      setResidents([]);
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    void refresh();
  }, [open, restBaseUrl]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) {
      setEnrollTarget(null);
      setEnrollName('');
      setRenameUuid(null);
      setConfirmDeleteUuid(null);
      setBanner(null);
    }
  }, [open]);

  // Drop enroll selection if that SID leaves the live set.
  useEffect(() => {
    if (!enrollTarget) return;
    const still = liveTracks.some((t) => Number(t.stable_id) === enrollTarget.stable_id);
    if (!still) {
      setEnrollTarget(null);
      setEnrollName('');
    }
  }, [liveTracks, enrollTarget]);

  const selectTrack = (t: HouseholdLiveTrack) => {
    const gate = canEnrollTrack(t);
    if (!gate.ok) {
      setBanner({ kind: 'info', text: gate.reason || 'Cannot enroll this track' });
      setEnrollTarget(null);
      return;
    }
    setBanner(null);
    setEnrollTarget({
      stable_id: Number(t.stable_id),
      camera_id: String(t.camera_id || ''),
      kind: kindOf(t),
      display_name: t.display_name,
    });
    setEnrollName('');
  };

  const submitEnroll = async () => {
    if (!enrollTarget) return;
    const name = enrollName.trim();
    if (!name) {
      setBanner({ kind: 'warn', text: 'Enter a display name' });
      return;
    }
    setBusy(true);
    setBanner(null);
    try {
      const sid = enrollTarget.stable_id;
      const body =
        classifyStableId(sid) === 'visitor'
          ? { display_name: name, visitor_id: sid }
          : { display_name: name, stable_id: sid };
      const resp = await enrollResident(restBaseUrl, body);
      setBanner({
        kind: 'ok',
        text: `Enrolled ${resp.resident.display_name} as resident #${resp.resident.stable_id}`,
      });
      setEnrollTarget(null);
      setEnrollName('');
      await refresh();
    } catch (err) {
      const detail = err instanceof HouseholdApiError ? err.detail : err instanceof Error ? err.message : 'Enroll failed';
      setBanner({ kind: 'error', text: detail });
    } finally {
      setBusy(false);
    }
  };

  const submitRename = async (uuid: string) => {
    const name = renameName.trim();
    if (!name) {
      setBanner({ kind: 'warn', text: 'Enter a display name' });
      return;
    }
    setBusy(true);
    setBanner(null);
    try {
      const resp = await patchResident(restBaseUrl, uuid, { display_name: name });
      setBanner({ kind: 'ok', text: `Renamed to ${resp.resident.display_name}` });
      setRenameUuid(null);
      setRenameName('');
      await refresh();
    } catch (err) {
      const detail = err instanceof HouseholdApiError ? err.detail : err instanceof Error ? err.message : 'Rename failed';
      setBanner({ kind: 'error', text: detail });
    } finally {
      setBusy(false);
    }
  };

  const submitDelete = async (uuid: string) => {
    setBusy(true);
    setBanner(null);
    try {
      const resp = await deleteResident(restBaseUrl, uuid);
      setBanner({
        kind: 'ok',
        text: `Removed ${resp.resident.display_name}. Live tracks (if any) remapped to a visitor.`,
      });
      setConfirmDeleteUuid(null);
      await refresh();
    } catch (err) {
      const detail = err instanceof HouseholdApiError ? err.detail : err instanceof Error ? err.message : 'Delete failed';
      setBanner({ kind: 'error', text: detail });
    } finally {
      setBusy(false);
    }
  };

  const applyMerge = async (c: SuggestCandidate) => {
    if (c.blocked) {
      setBanner({ kind: 'warn', text: c.block_reason || 'Merge blocked' });
      return;
    }
    setBusy(true);
    setBanner(null);
    try {
      const resp = await mergeAliases(restBaseUrl, {
        a: c.a,
        b: c.b,
        canonical: c.preferred_canonical || c.canonical,
        force: false,
      });
      if (resp.applied) {
        setBanner({ kind: 'ok', text: `Merged ${resp.src} → ${resp.canonical}` });
      } else {
        setBanner({ kind: 'warn', text: resp.reason || 'Merge not applied' });
      }
      await refresh();
    } catch (err) {
      const detail = err instanceof HouseholdApiError ? err.detail : err instanceof Error ? err.message : 'Merge failed';
      setBanner({ kind: 'error', text: detail });
    } finally {
      setBusy(false);
    }
  };

  // Dedupe live tracks by stable_id (prefer first camera seen).
  const uniqueTracks: HouseholdLiveTrack[] = [];
  const seen = new Set<number>();
  for (const t of liveTracks) {
    const sid = Number(t.stable_id);
    if (!Number.isFinite(sid) || seen.has(sid)) continue;
    seen.add(sid);
    uniqueTracks.push(t);
  }
  uniqueTracks.sort((a, b) => Number(a.stable_id) - Number(b.stable_id));

  const residentSids = new Set(residents.map((r) => Number(r.stable_id)));
  const visitorTracks = uniqueTracks.filter((t) => kindOf(t) === 'visitor' || classifyStableId(t.stable_id) === 'visitor');
  const provisionalTracks = uniqueTracks.filter((t) => kindOf(t) === 'provisional' || classifyStableId(t.stable_id) === 'provisional');
  const otherLive = uniqueTracks.filter((t) => {
    const k = kindOf(t);
    return k !== 'visitor' && k !== 'provisional' && !residentSids.has(Number(t.stable_id));
  });

  const trackColor = (t: HouseholdLiveTrack): string => {
    const camKey = detectCameraKey(t.camera_id) as CameraKey | null;
    const colorId = camKey ? colorIdForPerson(camKey, t.stable_id) : Number(t.stable_id || 0);
    return colorForTrack(colorId);
  };

  return (
    <>
      <div
        className={open ? 'hh-overlay open' : 'hh-overlay'}
        onClick={onClose}
        aria-hidden={!open}
      />
      <aside
        className={open ? 'hh-drawer open' : 'hh-drawer'}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-hidden={!open}
      >
        <div className="hh-header">
          <div>
            <h2 id={titleId}>People</h2>
            <div className="hh-subtitle">
              Enroll household residents, rename them, and watch identity health.
            </div>
          </div>
          <div className="hh-header-actions">
            <button
              type="button"
              className="btn ghost"
              disabled={loading || busy}
              onClick={() => void refresh()}
              title="Refresh"
            >
              {loading ? '…' : 'Refresh'}
            </button>
            <button type="button" className="hh-icon" aria-label="Close" title="Close" onClick={onClose}>
              ×
            </button>
          </div>
        </div>

        <div className="hh-content">
          {banner && <div className={`hh-banner ${banner.kind}`}>{banner.text}</div>}

          {householdOff && (
            <div className="hh-banner warn">
              Enrollment APIs require household mode. Live track names still update when the runtime is in household mode.
            </div>
          )}

          <div className="hh-stats">
            <div className="hh-stat">
              <div className="hh-stat-label">Residents</div>
              <div className="hh-stat-value">{health?.resident_count ?? residents.length}</div>
            </div>
            <div className="hh-stat">
              <div className="hh-stat-label">Visitors</div>
              <div className="hh-stat-value">{health?.visitor_count ?? '—'}</div>
            </div>
            <div className="hh-stat">
              <div className="hh-stat-label">Provisional</div>
              <div className="hh-stat-value">{health?.provisional_count ?? provisionalTracks.length}</div>
            </div>
            <div className="hh-stat">
              <div className="hh-stat-label">Active</div>
              <div className="hh-stat-value">{health?.active_unique ?? uniqueTracks.length}</div>
            </div>
          </div>

          <section className="hh-section">
            <div className="hh-section-head">
              <div className="hh-section-title">Live people</div>
              <div className="hh-section-meta">{uniqueTracks.length} unique ID{uniqueTracks.length === 1 ? '' : 's'}</div>
            </div>
            <p className="hh-hint">
              Select a <strong>visitor</strong> to enroll as a named resident. Prefer a clear, single-camera view
              (avoid kitchen↔family overlap when unsure who is who).
            </p>

            {!uniqueTracks.length && <div className="hh-empty">No active tracks right now.</div>}

            {visitorTracks.length > 0 && (
              <div className="hh-track-list">
                {visitorTracks.map((t) => {
                  const selected = enrollTarget?.stable_id === Number(t.stable_id);
                  const gate = canEnrollTrack(t);
                  return (
                    <button
                      key={`v-${t.stable_id}-${t.camera_id}`}
                      type="button"
                      className={`hh-track${selected ? ' selected' : ''}${gate.ok ? '' : ' disabled'}`}
                      onClick={() => selectTrack(t)}
                    >
                      <div className="hh-row">
                        <span className="hh-dot" style={{ background: trackColor(t) }} />
                        <span className="hh-name">Visitor {t.stable_id}</span>
                        <span className="spacer" />
                        <span className="hh-mono">{t.camera_id}</span>
                      </div>
                      <div className="hh-chips">
                        <span className="hh-chip visitor">visitor</span>
                        {t.zone ? <span className="hh-chip">{t.zone}</span> : null}
                        <span className="hh-chip">reid {confText(t.reid_confidence)}</span>
                        {t.embedding_present ? <span className="hh-chip">emb</span> : <span className="hh-chip warn">no emb</span>}
                        {t.overlap_permit ? <span className="hh-chip warn">overlap</span> : null}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {provisionalTracks.length > 0 && (
              <div className="hh-track-list">
                {provisionalTracks.map((t) => (
                  <div key={`p-${t.stable_id}`} className="hh-track disabled">
                    <div className="hh-row">
                      <span className="hh-dot" style={{ background: trackColor(t) }} />
                      <span className="hh-name">Provisional {t.stable_id}</span>
                      <span className="spacer" />
                      <span className="hh-mono">{t.camera_id}</span>
                    </div>
                    <div className="hh-chips">
                      <span className="hh-chip provisional">confirming…</span>
                      {t.zone ? <span className="hh-chip">{t.zone}</span> : null}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {otherLive.length > 0 && (
              <div className="hh-track-list">
                {otherLive.map((t) => (
                  <div key={`o-${t.stable_id}`} className="hh-track disabled">
                    <div className="hh-row">
                      <span className="hh-dot" style={{ background: trackColor(t) }} />
                      <span className="hh-name">{t.display_name || `ID ${t.stable_id}`}</span>
                      <span className="spacer" />
                      <span className="hh-mono">{t.camera_id}</span>
                    </div>
                    <div className="hh-chips">
                      <span className="hh-chip">{kindOf(t)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {enrollTarget && (
              <div className="hh-form">
                <div className="hh-label">
                  Enroll visitor <span className="hh-mono">{enrollTarget.stable_id}</span>
                  {enrollTarget.camera_id ? ` · ${enrollTarget.camera_id}` : ''}
                </div>
                <input
                  className="hh-input"
                  type="text"
                  autoFocus
                  placeholder="Display name (e.g. Alex)"
                  value={enrollName}
                  disabled={busy || householdOff}
                  onChange={(e) => setEnrollName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') void submitEnroll();
                  }}
                />
                <div className="hh-actions">
                  <button
                    type="button"
                    className="btn primary"
                    disabled={busy || householdOff || !enrollName.trim()}
                    onClick={() => void submitEnroll()}
                  >
                    {busy ? 'Enrolling…' : 'Enroll as resident'}
                  </button>
                  <button
                    type="button"
                    className="btn ghost"
                    disabled={busy}
                    onClick={() => {
                      setEnrollTarget(null);
                      setEnrollName('');
                    }}
                  >
                    Cancel
                  </button>
                </div>
                <div className="hh-hint">
                  Sticky resident ID is assigned server-side (1…N). Gallery embeddings move with the remap.
                </div>
              </div>
            )}
          </section>

          <section className="hh-section">
            <div className="hh-section-head">
              <div className="hh-section-title">Residents</div>
              <div className="hh-section-meta">{residents.length} enrolled</div>
            </div>
            {!residents.length && <div className="hh-empty">No residents enrolled yet.</div>}
            <div className="hh-resident-list">
              {residents.map((r) => {
                const live = uniqueTracks.find((t) => Number(t.stable_id) === Number(r.stable_id));
                const renaming = renameUuid === r.uuid;
                const confirming = confirmDeleteUuid === r.uuid;
                return (
                  <div key={r.uuid} className="hh-resident" style={{ cursor: 'default' }}>
                    <div className="hh-row">
                      <span
                        className="hh-dot"
                        style={{
                          background: live
                            ? trackColor(live)
                            : colorForTrack(Number(r.stable_id)),
                        }}
                      />
                      <span className="hh-name">{r.display_name}</span>
                      <span className="spacer" />
                      <span className="hh-mono">#{r.stable_id}</span>
                    </div>
                    <div className="hh-chips">
                      <span className="hh-chip resident">resident</span>
                      {live ? <span className="hh-chip">live · {live.camera_id}</span> : <span className="hh-chip">offline</span>}
                      <span className="hh-chip">
                        gallery {r.gallery_embeddings ?? r.embedding_count ?? 0}
                      </span>
                      <span className="hh-chip" title={r.uuid}>
                        {formatTs(r.created_ts)}
                      </span>
                    </div>

                    {renaming ? (
                      <div className="hh-form">
                        <input
                          className="hh-input"
                          type="text"
                          autoFocus
                          value={renameName}
                          disabled={busy}
                          onChange={(e) => setRenameName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') void submitRename(r.uuid);
                          }}
                        />
                        <div className="hh-actions">
                          <button
                            type="button"
                            className="btn primary"
                            disabled={busy || !renameName.trim()}
                            onClick={() => void submitRename(r.uuid)}
                          >
                            Save
                          </button>
                          <button
                            type="button"
                            className="btn ghost"
                            disabled={busy}
                            onClick={() => {
                              setRenameUuid(null);
                              setRenameName('');
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : confirming ? (
                      <div className="hh-form">
                        <div className="hh-hint">
                          Remove <strong>{r.display_name}</strong>? Live tracks remapped to a new visitor; gallery moves with them.
                        </div>
                        <div className="hh-actions">
                          <button
                            type="button"
                            className="btn danger"
                            disabled={busy}
                            onClick={() => void submitDelete(r.uuid)}
                          >
                            Confirm remove
                          </button>
                          <button
                            type="button"
                            className="btn ghost"
                            disabled={busy}
                            onClick={() => setConfirmDeleteUuid(null)}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="hh-resident-actions">
                        <button
                          type="button"
                          className="btn ghost"
                          disabled={busy || householdOff}
                          onClick={() => {
                            setRenameUuid(r.uuid);
                            setRenameName(r.display_name);
                            setConfirmDeleteUuid(null);
                          }}
                        >
                          Rename
                        </button>
                        <button
                          type="button"
                          className="btn danger"
                          disabled={busy || householdOff}
                          onClick={() => {
                            setConfirmDeleteUuid(r.uuid);
                            setRenameUuid(null);
                          }}
                        >
                          Remove
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          <details className="hh-section hh-details">
            <summary>
              <span>Identity health</span>
            </summary>
            <div className="hh-details-body">
              {!health && <div className="hh-empty">Health unavailable.</div>}
              {health && (
                <dl className="hh-health-grid">
                  <dt>False-share blocked</dt>
                  <dd>{health.false_share_blocked_count}</dd>
                  <dt>Overlap grants</dt>
                  <dd>{health.overlap_permit_grant_count}</dd>
                  <dt>Overlap denies</dt>
                  <dd>{health.overlap_permit_deny_count}</dd>
                  <dt>Visitor mints</dt>
                  <dd>{health.mint_visitor_count}</dd>
                  <dt>Resident promotes</dt>
                  <dd>{health.promote_resident_count}</dd>
                  <dt>Gallery IDs</dt>
                  <dd>{health.gallery_ids}</dd>
                  <dt>Quality rejects</dt>
                  <dd>{health.gallery_quality_reject_count}</dd>
                  <dt>MNN rejects</dt>
                  <dd>{health.mnn_reject_count}</dd>
                </dl>
              )}
            </div>
          </details>

          <details className="hh-section hh-details">
            <summary>
              <span>Suggested merges</span>
            </summary>
            <div className="hh-details-body">
              <p className="hh-hint">
                Household mode keeps auto-merge off. Review candidates; blocked pairs show why.
              </p>
              {!suggestions.length && <div className="hh-empty">No suggestions.</div>}
              <div className="hh-suggest-list">
                {suggestions.map((c) => (
                  <div
                    key={`${c.a}-${c.b}-${c.sim}`}
                    className={`hh-suggest${c.blocked ? ' blocked' : ''}`}
                  >
                    <div className="hh-row">
                      <span className="hh-name">
                        {c.a} ↔ {c.b}
                      </span>
                      <span className="spacer" />
                      <span className="hh-mono">sim {c.sim.toFixed(3)}</span>
                    </div>
                    <div className="hh-chips">
                      <span className="hh-chip">canonical {c.preferred_canonical || c.canonical}</span>
                      <span className="hh-chip">
                        emb {c.a_embedding_count}/{c.b_embedding_count}
                      </span>
                      {c.blocked ? (
                        <span className="hh-chip warn">{c.block_reason || 'blocked'}</span>
                      ) : (
                        <span className="hh-chip resident">ok</span>
                      )}
                    </div>
                    {!c.blocked && (
                      <div className="hh-actions">
                        <button
                          type="button"
                          className="btn ghost"
                          disabled={busy || householdOff}
                          onClick={() => void applyMerge(c)}
                        >
                          Merge manually
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </details>

          <p className="hh-footer-note">
            Residents persist under ~/.noesis/household/. Removing a resident does not delete video —
            it only unbinds the name and remaps the live SID to a visitor slot.
          </p>
        </div>
      </aside>
    </>
  );
};
