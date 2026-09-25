import {
  CircleAlert,
  CircleDot,
  LoaderCircle,
  RotateCcw,
} from 'lucide-react'
import {
  voiceUiStateLabel,
  type VoiceUiState,
} from '../voiceActivity'

type VoiceActivityIndicatorProps = {
  state: VoiceUiState
  disconnectedLabel?: string
}

function Waveform({ variant }: { variant: 'user' | 'agent' }) {
  return (
    <span
      className={`voice-waveform ${variant}-speaking`}
      aria-hidden="true"
    >
      {Array.from({ length: 6 }, (_, index) => (
        <i key={index} />
      ))}
    </span>
  )
}

export function VoiceActivityIndicator({
  state,
  disconnectedLabel = 'Disconnected',
}: VoiceActivityIndicatorProps) {
  const label = voiceUiStateLabel(state, disconnectedLabel)

  return (
    <span
      className={`voice-activity voice-activity-${state}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {state === 'user-speaking' ? (
        <Waveform variant="user" />
      ) : state === 'agent-speaking' ? (
        <Waveform variant="agent" />
      ) : state === 'processing' ? (
        <span className="voice-processing-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      ) : state === 'connecting' ? (
        <LoaderCircle className="voice-connecting-icon" size={14} aria-hidden="true" />
      ) : state === 'listening' ? (
        <span className="voice-listening-pulse" aria-hidden="true" />
      ) : state === 'muted' ? null : state === 'interrupted' ? (
        <RotateCcw size={14} aria-hidden="true" />
      ) : state === 'error' ? (
        <CircleAlert size={14} aria-hidden="true" />
      ) : (
        <CircleDot size={13} aria-hidden="true" />
      )}
      <span className="voice-activity-label">{label}</span>
    </span>
  )
}
