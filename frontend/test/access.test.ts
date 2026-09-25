import assert from 'node:assert/strict'
import test from 'node:test'

import { canStartVoice, canUseTextChat } from '../src/access.ts'

test('anonymous sessions can use text chat', () => {
  assert.equal(
    canUseTextChat({ sessionId: 'anonymous-session', processing: false }),
    true,
  )
})

test('text chat requires a session and pauses while processing', () => {
  assert.equal(canUseTextChat({ sessionId: null, processing: false }), false)
  assert.equal(
    canUseTextChat({ sessionId: 'anonymous-session', processing: true }),
    false,
  )
})

test('voice remains unavailable before authentication', () => {
  assert.equal(
    canStartVoice({
      sessionId: 'anonymous-session',
      authenticated: false,
      voiceConnectionState: 'disconnected',
    }),
    false,
  )
})

test('authentication keeps text enabled and enables disconnected voice', () => {
  assert.equal(
    canUseTextChat({ sessionId: 'authenticated-session', processing: false }),
    true,
  )
  assert.equal(
    canStartVoice({
      sessionId: 'authenticated-session',
      authenticated: true,
      voiceConnectionState: 'disconnected',
    }),
    true,
  )
})
