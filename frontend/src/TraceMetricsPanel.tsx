import {
  Activity,
  Check,
  CircleCheck,
  CircleDollarSign,
  Clock3,
  Copy,
  Database,
  FileText,
  MessageSquareText,
  RefreshCw,
  TriangleAlert,
  Wrench,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { useState } from 'react'
import {
  formatCost,
  formatLatency,
  latestTurn,
  readableStage,
  sortedLatencyStages,
  statusTone,
  type SessionObservabilityResponse,
  type StatusTone,
} from './traceMetrics'

type TraceMetricsPanelProps = {
  response: SessionObservabilityResponse | null
  loading: boolean
  error: string | null
  canRefresh: boolean
  onRefresh: () => void
}

type KpiCardProps = {
  icon: LucideIcon
  label: string
  value: string
  detail: string
  tone?: StatusTone
}

function KpiCard({
  icon: IconComponent,
  label,
  value,
  detail,
  tone = 'neutral',
}: KpiCardProps) {
  return (
    <article className={`trace-kpi-card ${tone}`}>
      <div className="trace-kpi-label">
        <span className="trace-kpi-icon">
          <IconComponent size={15} strokeWidth={1.8} aria-hidden="true" />
        </span>
        <span>{label}</span>
      </div>
      <strong title={value}>{value}</strong>
      <span className="trace-kpi-detail">{detail}</span>
    </article>
  )
}

function TraceIdentifier({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false)

  const copyValue = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    } catch {
      // The identifier remains selectable when clipboard access is unavailable.
    }
  }

  return (
    <div className="trace-detail-row">
      <dt>{label}</dt>
      <dd>
        <code title={value}>{value}</code>
        <button
          type="button"
          className="trace-copy-button"
          onClick={() => void copyValue()}
          aria-label={`Copy ${label}`}
          title={copied ? 'Copied' : `Copy ${label}`}
        >
          {copied ? (
            <Check size={13} aria-hidden="true" />
          ) : (
            <Copy size={13} aria-hidden="true" />
          )}
        </button>
      </dd>
    </div>
  )
}

export function TraceMetricsPanel({
  response,
  loading,
  error,
  canRefresh,
  onRefresh,
}: TraceMetricsPanelProps) {
  const latest = latestTurn(response)
  const session = response?.session ?? null
  const tracedTurnCount = response?.turns.length ?? 0
  const latencyStages = latest ? sortedLatencyStages(latest.latency_ms) : []
  const slowestLatency = Math.max(
    0,
    ...latencyStages.map(([, duration]) => duration),
  )
  const displayStatus = latest?.outcome ?? latest?.status ?? 'Unavailable'
  const tone = statusTone(displayStatus, latest?.errors)

  return (
    <aside className="trace-panel" aria-labelledby="trace-panel-heading">
      <header className="trace-panel-header">
        <div>
          <div className="trace-title-line">
            <p className="section-label">Reviewer inspector</p>
            <span className="trace-demo-badge">Demo</span>
          </div>
          <h2 id="trace-panel-heading">Trace &amp; Metrics</h2>
          <p>Reviewer view of agent traces, tools, retrieval and latency.</p>
        </div>
        <button
          type="button"
          className="trace-refresh-button"
          onClick={onRefresh}
          disabled={!canRefresh || loading}
          aria-label="Refresh trace metrics"
          title="Refresh trace metrics"
        >
          <RefreshCw
            size={15}
            className={loading ? 'is-refreshing' : undefined}
            aria-hidden="true"
          />
        </button>
      </header>

      <div className="trace-panel-content">
        {error && (
          <div className="trace-fetch-error" role="alert">
            <TriangleAlert size={15} aria-hidden="true" />
            <div>
              <strong>Metrics refresh failed</strong>
              <span>{error}</span>
            </div>
          </div>
        )}

        {loading && !response ? (
          <div className="trace-panel-state" role="status">
            <Activity size={22} aria-hidden="true" />
            <strong>Loading trace summary…</strong>
            <span>The banking conversation remains available.</span>
          </div>
        ) : !session || !latest ? (
          <div className="trace-panel-state trace-empty">
            <Activity size={22} aria-hidden="true" />
            <strong>No traced turns yet</strong>
            <span>Complete a text or voice turn to inspect execution.</span>
          </div>
        ) : (
          <div className="trace-panel-layout">
            <section aria-label="Session key performance indicators">
              <div className="trace-kpi-grid">
                <KpiCard
                  icon={MessageSquareText}
                  label="Turns"
                  value={session.turn_count.toLocaleString('en-US')}
                  detail="Traced in this session"
                />
                <KpiCard
                  icon={CircleDollarSign}
                  label="Est. cost"
                  value={formatCost(
                    session.estimated_total_cost_usd,
                    session.cost_unavailable,
                  )}
                  detail="Provider estimate"
                />
                <KpiCard
                  icon={Clock3}
                  label="Agent turn"
                  value={formatLatency(latest.latency_ms['agent.turn'])}
                  detail="Latest measured turn"
                />
                <KpiCard
                  icon={
                    tone === 'danger'
                      ? TriangleAlert
                      : tone === 'success'
                        ? CircleCheck
                        : Activity
                  }
                  label="Status"
                  value={displayStatus.replaceAll('_', ' ')}
                  detail="Latest turn outcome"
                  tone={tone}
                />
              </div>
            </section>

            <section className="trace-section trace-latest-turn">
              <div className="trace-section-heading">
                <div>
                  <p className="section-label">Latest turn</p>
                  <h3>
                    Turn {tracedTurnCount} of {session.turn_count}
                  </h3>
                </div>
                <span className={`trace-outcome ${tone}`}>
                  {displayStatus}
                </span>
              </div>

              <dl className="trace-detail-list">
                <TraceIdentifier label="Trace ID" value={latest.trace_id} />
                <TraceIdentifier label="Turn ID" value={latest.turn_id} />
              </dl>
            </section>

            <div className="trace-detail-stack">
              <section
                className="trace-section trace-mini-card"
                aria-labelledby="trace-tools-heading"
              >
              <div className="trace-section-heading compact">
                <div className="trace-section-icon">
                  <Wrench size={14} aria-hidden="true" />
                </div>
                <h3 id="trace-tools-heading">Tool calls</h3>
                <span className="trace-count">{latest.tool_calls.length}</span>
              </div>
              {latest.tool_calls.length > 0 ? (
                <div className="trace-chip-list">
                  {latest.tool_calls.map((tool, index) => (
                    <span className="trace-chip tool" key={`${tool}-${index}`}>
                      {tool}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="trace-none">No tool executed</p>
              )}
              </section>

              <section
                className="trace-section trace-mini-card"
                aria-labelledby="trace-rag-heading"
              >
              <div className="trace-section-heading compact">
                <div className="trace-section-icon">
                  <Database size={14} aria-hidden="true" />
                </div>
                <h3 id="trace-rag-heading">Retrieval</h3>
                <span className="trace-count">
                  {latest.retrieval_count}{' '}
                  {latest.retrieval_count === 1 ? 'document' : 'documents'}
                </span>
              </div>
              {latest.retrieval_count === 0 ? (
                <p className="trace-none">No policy retrieval</p>
              ) : latest.policy_sources.length > 0 ? (
                <>
                  <p className="trace-sub-label">
                    <FileText size={12} aria-hidden="true" /> Policy sources
                  </p>
                  <div className="trace-chip-list">
                    {latest.policy_sources.map((source) => (
                      <span className="trace-chip" key={source} title={source}>
                        {source}
                      </span>
                    ))}
                  </div>
                </>
              ) : (
                <p className="trace-none">No policy sources recorded</p>
              )}
              </section>

              <section
                className="trace-section trace-mini-card trace-latency-section"
                aria-labelledby="trace-latency-heading"
              >
              <div className="trace-section-heading compact">
                <div className="trace-section-icon">
                  <Clock3 size={14} aria-hidden="true" />
                </div>
                <h3 id="trace-latency-heading">Latency</h3>
                <span className="trace-count">Latest turn</span>
              </div>
              {latencyStages.length > 0 ? (
                <div className="trace-latency-list">
                  {latencyStages.map(([stage, duration]) => {
                    const width =
                      slowestLatency === 0
                        ? 0
                        : Math.max(2, (duration / slowestLatency) * 100)
                    return (
                      <div className="trace-latency-row" key={stage}>
                        <div className="trace-latency-meta">
                          <span>{readableStage(stage)}</span>
                          <strong>{formatLatency(duration)}</strong>
                        </div>
                        <div className="trace-latency-track" aria-hidden="true">
                          <span
                            className="trace-latency-fill"
                            style={{ width: `${width}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <p className="trace-none">No latency stages recorded</p>
              )}
              </section>

              <section
                className="trace-section trace-mini-card trace-errors-section"
                aria-labelledby="trace-errors-heading"
              >
              <div className="trace-section-heading compact">
                <div className="trace-section-icon">
                  <TriangleAlert size={14} aria-hidden="true" />
                </div>
                <h3 id="trace-errors-heading">Errors</h3>
              </div>
              {latest.errors.length === 0 ? (
                <div className="trace-errors success">
                  <CircleCheck size={15} aria-hidden="true" />
                  <div>
                    <strong>None</strong>
                    <span>No errors in this turn.</span>
                  </div>
                </div>
              ) : (
                <div className="trace-errors danger">
                  <TriangleAlert size={15} aria-hidden="true" />
                  <div>
                    <strong>{latest.errors.length} safe error categories</strong>
                    <ul>
                      {latest.errors.map((category, index) => (
                        <li key={`${category}-${index}`}>{category}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              </section>
            </div>
          </div>
        )}
      </div>

      <footer className="trace-footer">
        FastAPI process trace <span aria-hidden="true">·</span> demo scope
      </footer>
    </aside>
  )
}
