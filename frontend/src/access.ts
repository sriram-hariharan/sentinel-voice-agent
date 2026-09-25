export type VoiceConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'

export function canUseTextChat({
  sessionId,
  processing,
}: {
  sessionId: string | null
  processing: boolean
}): boolean {
  return sessionId !== null && !processing
}

export function canStartVoice({
  sessionId,
  authenticated,
  voiceConnectionState,
}: {
  sessionId: string | null
  authenticated: boolean
  voiceConnectionState: VoiceConnectionState
}): boolean {
  return (
    sessionId !== null &&
    authenticated &&
    voiceConnectionState === 'disconnected'
  )
}
