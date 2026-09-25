import {
  CircleCheck,
  LogIn,
  MessageCircle,
  Mic,
  MicOff,
  ShieldCheck,
  Square,
  UserRound,
} from 'lucide-react'
import type { FormEvent, ReactNode } from 'react'
import {
  microphoneControlPresentation,
  type VoiceUiState,
} from '../voiceActivity'
import { VoiceActivityIndicator } from './VoiceActivityIndicator'

type SessionControlBarProps = {
  authenticated: boolean
  authenticating: boolean
  email: string
  pin: string
  sessionId: string | null
  sessionStarting: boolean
  conversationPhase: string
  turnStatus: string
  pendingAction: string | null
  voiceUiState: VoiceUiState
  voiceError: string | null
  voiceStartEnabled: boolean
  voiceActive: boolean
  microphoneEnabled: boolean
  microphoneToggleEnabled: boolean
  microphoneTogglePending: boolean
  onEmailChange: (value: string) => void
  onPinChange: (value: string) => void
  onSignIn: (event: FormEvent<HTMLFormElement>) => void
  onStartVoice: () => void
  onEndVoice: () => void
  onToggleMicrophone: () => void
  voiceControls?: ReactNode
}

function formatState(value: string): string {
  return value.toLowerCase().replaceAll('_', ' ')
}

function ControlIcon({ children }: { children: ReactNode }) {
  return <span className="session-control-icon">{children}</span>
}

export function SessionControlBar({
  authenticated,
  authenticating,
  email,
  pin,
  sessionId,
  sessionStarting,
  conversationPhase,
  turnStatus,
  pendingAction,
  voiceUiState,
  voiceError,
  voiceStartEnabled,
  voiceActive,
  microphoneEnabled,
  microphoneToggleEnabled,
  microphoneTogglePending,
  onEmailChange,
  onPinChange,
  onSignIn,
  onStartVoice,
  onEndVoice,
  onToggleMicrophone,
  voiceControls,
}: SessionControlBarProps) {
  const microphoneControl = microphoneControlPresentation({
    connected: microphoneToggleEnabled,
    microphoneEnabled,
    pending: microphoneTogglePending,
  })

  return (
    <section
      className={`session-control-bar ${
        authenticated ? 'authenticated' : 'unauthenticated'
      }`}
      aria-label="Session and voice controls"
      title={sessionId ? `Session ${sessionId}` : undefined}
    >
      {authenticated ? (
        <>
          <div className="session-control-item session-control-identity">
            <ControlIcon>
              <UserRound size={18} aria-hidden="true" />
            </ControlIcon>
            <div>
              <span>Signed in as</span>
              <strong title={email}>{email || 'Synthetic customer'}</strong>
            </div>
          </div>

          <div className="session-control-item session-control-authenticated">
            <ControlIcon>
              <ShieldCheck size={18} aria-hidden="true" />
            </ControlIcon>
            <div>
              <span>Authentication</span>
              <strong>
                Authenticated <i className="session-status-dot success" />
              </strong>
            </div>
          </div>
        </>
      ) : (
        <form className="session-auth-form" onSubmit={onSignIn}>
          <ControlIcon>
            <UserRound size={19} aria-hidden="true" />
          </ControlIcon>
          <div className="session-auth-content">
            <div className="session-auth-heading">
              <span>Secure access</span>
              <strong>Demo sign in</strong>
            </div>
            <div className="session-auth-fields">
              <label className="session-auth-field" htmlFor="session-email">
                <span>Synthetic customer email</span>
                <input
                  id="session-email"
                  type="email"
                  value={email}
                  onChange={(event) => onEmailChange(event.target.value)}
                  placeholder="name@example.test"
                  autoComplete="username"
                  required
                  disabled={authenticating}
                />
              </label>
              <label className="session-auth-field" htmlFor="session-pin">
                <span>Demo PIN</span>
                <input
                  id="session-pin"
                  type="password"
                  value={pin}
                  onChange={(event) => onPinChange(event.target.value)}
                  placeholder="••••"
                  autoComplete="current-password"
                  required
                  disabled={authenticating}
                />
              </label>
              <button
                type="submit"
                className="session-sign-in-button"
                disabled={sessionStarting || authenticating}
              >
                <LogIn size={15} aria-hidden="true" />
                {sessionStarting || authenticating ? 'Starting…' : 'Sign In'}
              </button>
            </div>
          </div>
        </form>
      )}

      <div className="session-control-item session-control-state">
        <ControlIcon>
          <MessageCircle size={18} aria-hidden="true" />
        </ControlIcon>
        <div>
          <span>Session state</span>
          <strong>{formatState(conversationPhase)}</strong>
        </div>
      </div>

      <div className="session-control-item session-control-voice">
        <button
          type="button"
          className={`session-control-icon session-microphone-toggle${
            microphoneControl.muted ? ' muted' : ''
          }`}
          onClick={onToggleMicrophone}
          disabled={microphoneControl.disabled}
          aria-label={microphoneControl.label}
          title={microphoneControl.label}
          aria-pressed={microphoneControl.muted}
        >
          {microphoneControl.muted ? (
            <MicOff size={18} aria-hidden="true" />
          ) : (
            <Mic size={18} aria-hidden="true" />
          )}
        </button>
        <div className="session-control-voice-info">
          <span>Voice session</span>
          <span title={voiceError ?? undefined}>
            <VoiceActivityIndicator
              state={voiceUiState}
              disconnectedLabel={
                authenticated ? 'Ready to connect' : 'Disconnected'
              }
            />
          </span>
        </div>
        {voiceControls && (
          <div className="session-control-runtime">{voiceControls}</div>
        )}
        <div className="session-voice-actions">
          <button
            type="button"
            className="session-start-voice"
            onClick={onStartVoice}
            disabled={!voiceStartEnabled}
          >
            {voiceUiState === 'connecting' ? 'Connecting…' : 'Start Voice'}
          </button>
          <button
            type="button"
            className="session-end-voice"
            onClick={onEndVoice}
            disabled={!voiceActive}
            aria-label="End voice session"
            title="End voice session"
          >
            <Square size={13} fill="currentColor" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="session-control-item session-control-turn">
        <ControlIcon>
          <CircleCheck size={18} aria-hidden="true" />
        </ControlIcon>
        <div>
          <span>Turn status</span>
          <strong title={pendingAction ?? undefined}>
            <i
              className={`session-status-dot ${
                turnStatus === 'FAILED'
                  ? 'danger'
                  : turnStatus === 'PROCESSING' ||
                      turnStatus === 'WAITING_FOR_CONFIRMATION'
                    ? 'warning'
                    : 'success'
              }`}
            />
            {formatState(turnStatus)}
          </strong>
        </div>
      </div>
    </section>
  )
}
