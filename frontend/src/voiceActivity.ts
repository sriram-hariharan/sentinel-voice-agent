import type { VoiceConnectionState } from './access'

export type VoiceUiState =
  | 'disconnected'
  | 'connecting'
  | 'muted'
  | 'listening'
  | 'user-speaking'
  | 'processing'
  | 'agent-speaking'
  | 'interrupted'
  | 'error'

type VoiceUiStateInput = {
  voiceConnectionState: VoiceConnectionState
  voiceAgentState: string
  microphoneMuted: boolean
  userSpeaking: boolean
  voiceTurnPending: boolean
  voiceInterrupted: boolean
  voiceError: string | null
}

export function deriveVoiceUiState({
  voiceConnectionState,
  voiceAgentState,
  microphoneMuted,
  userSpeaking,
  voiceTurnPending,
  voiceInterrupted,
  voiceError,
}: VoiceUiStateInput): VoiceUiState {
  if (voiceError) {
    return 'error'
  }

  if (voiceConnectionState === 'connecting') {
    return 'connecting'
  }

  if (voiceConnectionState === 'disconnected') {
    return 'disconnected'
  }

  if (microphoneMuted) {
    return 'muted'
  }

  if (voiceAgentState === 'speaking') {
    return 'agent-speaking'
  }

  if (userSpeaking) {
    return 'user-speaking'
  }

  if (voiceTurnPending || voiceAgentState === 'thinking') {
    return 'processing'
  }

  if (voiceInterrupted) {
    return 'interrupted'
  }

  return 'listening'
}

export function voiceUiStateLabel(
  state: VoiceUiState,
  disconnectedLabel = 'Disconnected',
): string {
  const labels: Record<Exclude<VoiceUiState, 'disconnected'>, string> = {
    connecting: 'Connecting',
    muted: 'Muted',
    listening: 'Listening',
    'user-speaking': "You're speaking",
    processing: 'Processing',
    'agent-speaking': 'SentinelVoice speaking',
    interrupted: 'Interrupted',
    error: 'Voice error',
  }

  return state === 'disconnected' ? disconnectedLabel : labels[state]
}

export type MicrophoneController = (enabled: boolean) => Promise<boolean>

type MicrophoneToggleInput = {
  connected: boolean
  microphoneEnabled: boolean
  controller: MicrophoneController | null
}

export async function requestMicrophoneToggle({
  connected,
  microphoneEnabled,
  controller,
}: MicrophoneToggleInput): Promise<boolean | null> {
  if (!connected || !controller) {
    return null
  }

  return controller(!microphoneEnabled)
}

export function microphoneControlPresentation({
  connected,
  microphoneEnabled,
  pending = false,
}: {
  connected: boolean
  microphoneEnabled: boolean
  pending?: boolean
}) {
  const muted = connected && !microphoneEnabled

  return {
    muted,
    disabled: !connected || pending,
    label: connected
      ? muted
        ? 'Unmute microphone'
        : 'Mute microphone'
      : 'Microphone unavailable',
  }
}
