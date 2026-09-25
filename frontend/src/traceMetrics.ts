export type NumericSummary = {
  count: number
  minimum: number | null
  maximum: number | null
  mean: number | null
  p50: number | null
  p90: number | null
  p95: number | null
}

export type UsageRecord = {
  provider: string
  model: string
  operation: string
  quantities: Record<string, number>
}

export type TurnTraceSummary = {
  trace_id: string
  turn_id: string
  status: string
  outcome: string | null
  tool_calls: string[]
  retrieval_count: number
  policy_sources: string[]
  latency_ms: Record<string, number>
  usage: UsageRecord[]
  estimated_cost_usd: string | null
  cost_unavailable: string[]
  errors: string[]
}

export type SessionTraceSummary = {
  session_id: string
  turn_count: number
  successful_tasks: number | null
  latency: Record<string, NumericSummary>
  usage: UsageRecord[]
  estimated_total_cost_usd: string | null
  estimated_cost_per_success_usd: string | null
  cost_unavailable: string[]
}

export type SessionObservabilityResponse = {
  session: SessionTraceSummary | null
  turns: TurnTraceSummary[]
}

export type StatusTone = 'success' | 'warning' | 'danger' | 'neutral'

const preferredLatencyOrder = [
  'agent.turn',
  'voice.backend_turn',
  'stt',
  'rag.retrieval',
  'llm.request',
  'tool.execution',
  'tts.first_audio',
  'tts',
  'voice.interruption',
]

const preferredLatencyIndex = new Map(
  preferredLatencyOrder.map((stage, index) => [stage, index]),
)

export function latestTurn(
  response: SessionObservabilityResponse | null,
): TurnTraceSummary | null {
  return response?.turns.at(-1) ?? null
}

export function formatCost(
  value: string | null,
  unavailable: string[] = [],
): string {
  if (value === null || unavailable.length > 0) {
    return 'Unavailable'
  }

  const numericValue = Number(value)
  if (!Number.isFinite(numericValue)) {
    return 'Unavailable'
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumSignificantDigits: 4,
  }).format(numericValue)
}

export function formatLatency(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'Unavailable'
  }
  if (value === 0) {
    return '0 ms'
  }
  if (value < 1) {
    return `${value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')} ms`
  }
  if (value < 100) {
    return `${value.toFixed(1).replace(/\.0$/, '')} ms`
  }
  return `${Math.round(value).toLocaleString('en-US')} ms`
}

export function readableStage(stage: string): string {
  const labels: Record<string, string> = {
    'agent.turn': 'Agent turn',
    'voice.backend_turn': 'Voice backend',
    stt: 'Speech to text',
    'rag.retrieval': 'Policy retrieval',
    'llm.request': 'LLM request',
    'tool.execution': 'Tool execution',
    'tts.first_audio': 'TTS first audio',
    tts: 'Text to speech',
    'voice.interruption': 'Interruption stop',
  }

  return (
    labels[stage] ??
    stage
      .split('.')
      .map((part) => part.replaceAll('_', ' '))
      .join(' · ')
  )
}

export function sortedLatencyStages(
  latency: Record<string, number>,
): Array<[string, number]> {
  return Object.entries(latency).sort(([left], [right]) => {
    const leftIndex = preferredLatencyIndex.get(left)
    const rightIndex = preferredLatencyIndex.get(right)

    if (leftIndex !== undefined || rightIndex !== undefined) {
      return (
        (leftIndex ?? preferredLatencyOrder.length) -
        (rightIndex ?? preferredLatencyOrder.length)
      )
    }
    return left.localeCompare(right)
  })
}

export function statusTone(
  status: string | null | undefined,
  errors: string[] = [],
): StatusTone {
  const normalized = status?.toLowerCase() ?? ''
  if (
    errors.length > 0 ||
    normalized.includes('fail') ||
    normalized.includes('error')
  ) {
    return 'danger'
  }
  if (
    normalized.includes('waiting') ||
    normalized.includes('pending') ||
    normalized.includes('processing') ||
    normalized.includes('started')
  ) {
    return 'warning'
  }
  if (
    normalized.includes('completed') ||
    normalized.includes('responded') ||
    normalized.includes('confirmed')
  ) {
    return 'success'
  }
  return 'neutral'
}
