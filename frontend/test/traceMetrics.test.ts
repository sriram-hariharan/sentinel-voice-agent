import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatCost,
  formatLatency,
  latestTurn,
  readableStage,
  sortedLatencyStages,
  statusTone,
  type SessionObservabilityResponse,
  type TurnTraceSummary,
} from '../src/traceMetrics.ts'

const successfulTurn: TurnTraceSummary = {
  trace_id: 'trace_1234567890123456',
  turn_id: 'turn_12345678901234567',
  status: 'completed',
  outcome: 'RESPONDED',
  tool_calls: ['get_account_balance'],
  retrieval_count: 2,
  policy_sources: ['deposit-availability', 'account-access'],
  latency_ms: {
    'tool.execution': 4.2,
    'agent.turn': 18.4,
    'llm.request': 9.25,
  },
  usage: [],
  estimated_cost_usd: '0.0002565000',
  cost_unavailable: [],
  errors: [],
}

test('empty observability response has no latest turn', () => {
  assert.equal(latestTurn({ session: null, turns: [] }), null)
  assert.equal(latestTurn(null), null)
})

test('latest successful turn retains tools and policy sources', () => {
  const response = {
    session: null,
    turns: [{ ...successfulTurn, turn_id: 'older-turn' }, successfulTurn],
  } satisfies SessionObservabilityResponse

  const latest = latestTurn(response)
  assert.equal(latest?.outcome, 'RESPONDED')
  assert.deepEqual(latest?.tool_calls, ['get_account_balance'])
  assert.deepEqual(latest?.policy_sources, [
    'deposit-availability',
    'account-access',
  ])
  assert.equal(statusTone(latest?.outcome, latest?.errors), 'success')
})

test('no-tool and no-retrieval data remain empty rather than inferred', () => {
  const turn = {
    ...successfulTurn,
    tool_calls: [],
    retrieval_count: 0,
    policy_sources: [],
  }

  assert.deepEqual(turn.tool_calls, [])
  assert.equal(turn.retrieval_count, 0)
  assert.deepEqual(turn.policy_sources, [])
})

test('unavailable costs are never rendered as zero', () => {
  assert.equal(formatCost(null), 'Unavailable')
  assert.equal(
    formatCost('0', ['unknown/provider/operation/unit']),
    'Unavailable',
  )
  assert.equal(formatCost('0.0002565000'), '$0.0002565')
})

test('errors override a completed status tone', () => {
  assert.equal(statusTone('completed', ['tool_error']), 'danger')
  assert.equal(statusTone('WAITING_FOR_CONFIRMATION'), 'warning')
})

test('latency stages use preferred order and readable formatting', () => {
  assert.deepEqual(
    sortedLatencyStages(successfulTurn.latency_ms).map(([stage]) => stage),
    ['agent.turn', 'llm.request', 'tool.execution'],
  )
  assert.equal(readableStage('rag.retrieval'), 'Policy retrieval')
  assert.equal(readableStage('custom.stage_name'), 'custom · stage name')
  assert.equal(formatLatency(0), '0 ms')
  assert.equal(formatLatency(0.42), '0.42 ms')
  assert.equal(formatLatency(18.4), '18.4 ms')
  assert.equal(formatLatency(undefined), 'Unavailable')
})
