import assert from 'node:assert/strict'
import test from 'node:test'

import {
  deriveVoiceUiState,
  microphoneControlPresentation,
  requestMicrophoneToggle,
  voiceUiStateLabel,
} from '../src/voiceActivity.ts'

const connectedIdle = {
  voiceConnectionState: 'connected' as const,
  voiceAgentState: 'idle',
  microphoneMuted: false,
  userSpeaking: false,
  voiceTurnPending: false,
  voiceInterrupted: false,
  voiceError: null,
}

test('derives every supported voice presentation state', () => {
  assert.equal(
    deriveVoiceUiState({
      ...connectedIdle,
      voiceConnectionState: 'disconnected',
    }),
    'disconnected',
  )
  assert.equal(
    deriveVoiceUiState({
      ...connectedIdle,
      voiceConnectionState: 'connecting',
    }),
    'connecting',
  )
  assert.equal(deriveVoiceUiState(connectedIdle), 'listening')
  assert.equal(
    deriveVoiceUiState({ ...connectedIdle, userSpeaking: true }),
    'user-speaking',
  )
  assert.equal(
    deriveVoiceUiState({ ...connectedIdle, voiceTurnPending: true }),
    'processing',
  )
  assert.equal(
    deriveVoiceUiState({ ...connectedIdle, voiceAgentState: 'thinking' }),
    'processing',
  )
  assert.equal(
    deriveVoiceUiState({ ...connectedIdle, voiceAgentState: 'speaking' }),
    'agent-speaking',
  )
  assert.equal(
    deriveVoiceUiState({ ...connectedIdle, microphoneMuted: true }),
    'muted',
  )
  assert.equal(
    deriveVoiceUiState({ ...connectedIdle, voiceInterrupted: true }),
    'interrupted',
  )
  assert.equal(
    deriveVoiceUiState({ ...connectedIdle, voiceError: 'device failed' }),
    'error',
  )
})

test('speaking and processing precedence avoids contradictory states', () => {
  assert.equal(
    deriveVoiceUiState({
      ...connectedIdle,
      microphoneMuted: true,
      voiceAgentState: 'speaking',
      userSpeaking: true,
      voiceTurnPending: true,
    }),
    'muted',
  )
  assert.equal(
    deriveVoiceUiState({
      ...connectedIdle,
      voiceAgentState: 'speaking',
      userSpeaking: true,
      voiceTurnPending: true,
      voiceInterrupted: true,
    }),
    'agent-speaking',
  )
  assert.equal(
    deriveVoiceUiState({
      ...connectedIdle,
      userSpeaking: true,
      voiceTurnPending: true,
      voiceInterrupted: true,
    }),
    'user-speaking',
  )
  assert.equal(
    deriveVoiceUiState({
      ...connectedIdle,
      voiceTurnPending: true,
      voiceInterrupted: true,
    }),
    'processing',
  )
})

test('voice state labels remain stable and explicit', () => {
  assert.equal(voiceUiStateLabel('muted'), 'Muted')
  assert.equal(voiceUiStateLabel('user-speaking'), "You're speaking")
  assert.equal(
    voiceUiStateLabel('agent-speaking'),
    'SentinelVoice speaking',
  )
  assert.equal(voiceUiStateLabel('processing'), 'Processing')
  assert.equal(
    voiceUiStateLabel('disconnected', 'Ready to connect'),
    'Ready to connect',
  )
})

test('microphone toggle is inert until a LiveKit controller is connected', async () => {
  let calls = 0
  const result = await requestMicrophoneToggle({
    connected: false,
    microphoneEnabled: true,
    controller: async () => {
      calls += 1
      return false
    },
  })

  assert.equal(result, null)
  assert.equal(calls, 0)
  assert.deepEqual(
    microphoneControlPresentation({
      connected: false,
      microphoneEnabled: true,
    }),
    {
      muted: false,
      disabled: true,
      label: 'Microphone unavailable',
    },
  )
})

test('microphone toggle requests the inverse authoritative state', async () => {
  const requested: boolean[] = []
  const controller = async (enabled: boolean) => {
    requested.push(enabled)
    return enabled
  }

  assert.equal(
    await requestMicrophoneToggle({
      connected: true,
      microphoneEnabled: true,
      controller,
    }),
    false,
  )
  assert.deepEqual(
    microphoneControlPresentation({
      connected: true,
      microphoneEnabled: false,
    }),
    {
      muted: true,
      disabled: false,
      label: 'Unmute microphone',
    },
  )

  assert.equal(
    await requestMicrophoneToggle({
      connected: true,
      microphoneEnabled: false,
      controller,
    }),
    true,
  )
  assert.deepEqual(requested, [false, true])
  assert.deepEqual(
    microphoneControlPresentation({
      connected: true,
      microphoneEnabled: true,
    }),
    {
      muted: false,
      disabled: false,
      label: 'Mute microphone',
    },
  )
})
