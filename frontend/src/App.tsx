import {
  LiveKitRoom,
  RoomAudioRenderer,
  StartAudio,
  VoiceAssistantControlBar,
  useDataChannel,
  useLocalParticipant,
  useTranscriptions,
  useVoiceAssistant,
} from '@livekit/components-react'
import '@livekit/components-styles'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import './App.css'
import {
  canStartVoice,
  canUseTextChat,
  type VoiceConnectionState,
} from './access'

type SessionResponse = {
  session_id: string
  customer_id: string | null
  authenticated: boolean
  conversation_phase: string
  turn_status: string | null
  pending_action: string | null
  policy_sources: string[]
  voice_playback?: {
    speech_id: string
    voice_turn_id: string
    sequence: number
    status: 'SCHEDULED' | 'COMPLETED' | 'INTERRUPTED'
    interruption_stop_latency_ms: number | null
  } | null
}

type MessageResponse = Omit<
  SessionResponse,
  'turn_status' | 'voice_playback'
> & {
  message: string
  turn_status: string
  executed_tools: string[]
}

type TranscriptMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
  interrupted?: boolean
  sources?: string[]
}

type VoiceInterruptionEvent = {
  type: 'speech_interrupted'
  speech_id: string
  voice_turn_id: string
  interruption_stop_latency_ms: number
}

type VoiceConnectionToken = {
  server_url: string
  participant_token: string
}

function newTranscriptMessage(
  role: TranscriptMessage['role'],
  text: string,
  sources: string[] = [],
): TranscriptMessage {
  return { id: crypto.randomUUID(), role, text, sources }
}

async function postJson<T>(path: string, body?: object): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })

  const payload: unknown = await response.json().catch(() => null)

  if (!response.ok) {
    const detail =
      payload &&
      typeof payload === 'object' &&
      'detail' in payload &&
      typeof payload.detail === 'string'
        ? payload.detail
        : 'The request could not be completed.'

    throw new Error(detail)
  }

  return payload as T
}

let initialSessionRequest: Promise<SessionResponse> | null = null

function createInitialSession(): Promise<SessionResponse> {
  initialSessionRequest ??= postJson<SessionResponse>('/sessions').catch(
    (error: unknown) => {
      initialSessionRequest = null
      throw error
    },
  )
  return initialSessionRequest
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  const payload: unknown = await response.json().catch(() => null)

  if (!response.ok) {
    const detail =
      payload &&
      typeof payload === 'object' &&
      'detail' in payload &&
      typeof payload.detail === 'string'
        ? payload.detail
        : 'The request could not be completed.'

    throw new Error(detail)
  }

  return payload as T
}

function formatState(value: string): string {
  return value.toLowerCase().replaceAll('_', ' ')
}

type VoiceRoomProps = {
  onAgentStateChange: (state: string) => void
  onAgentFinalTranscript: () => void
  onError: (message: string) => void
  onInterruption: (event: VoiceInterruptionEvent) => void
  onTranscript: (message: TranscriptMessage) => void
  onUserFinalTranscript: () => void
}

function VoiceRoom({
  onAgentStateChange,
  onAgentFinalTranscript,
  onError,
  onInterruption,
  onTranscript,
  onUserFinalTranscript,
}: VoiceRoomProps) {
  const transcriptions = useTranscriptions()
  const { localParticipant, microphoneTrack } = useLocalParticipant()
  const { state: agentState } = useVoiceAssistant()
  const seenFinalSegments = useRef(new Set<string>())

  useDataChannel('sentinelvoice.voice', ({ payload }) => {
    try {
      const event: unknown = JSON.parse(new TextDecoder().decode(payload))

      if (
        event &&
        typeof event === 'object' &&
        'type' in event &&
        event.type === 'speech_interrupted' &&
        'speech_id' in event &&
        typeof event.speech_id === 'string' &&
        'voice_turn_id' in event &&
        typeof event.voice_turn_id === 'string' &&
        'interruption_stop_latency_ms' in event &&
        typeof event.interruption_stop_latency_ms === 'number'
      ) {
        onInterruption(event as VoiceInterruptionEvent)
      }
    } catch {
      // Ignore unrelated or malformed room data without disrupting audio.
    }
  })

  useEffect(() => {
    for (const transcription of transcriptions) {
      const attributes = transcription.streamInfo.attributes ?? {}

      if (attributes['lk.transcription_final'] !== 'true') {
        continue
      }

      const segmentId = attributes['lk.segment_id'] ?? transcription.streamInfo.id

      if (seenFinalSegments.current.has(segmentId)) {
        continue
      }

      const text = transcription.text.trim()

      if (!text) {
        continue
      }

      seenFinalSegments.current.add(segmentId)
      const role =
        transcription.participantInfo.identity === localParticipant.identity
          ? 'user'
          : 'assistant'

      onTranscript({ id: `voice-${segmentId}`, role, text })

      if (role === 'user') {
        onUserFinalTranscript()
      } else {
        if (text.startsWith('Voice session error:')) {
          onError(text)
        }
        onAgentFinalTranscript()
      }
    }
  }, [
    localParticipant.identity,
    onError,
    onAgentFinalTranscript,
    onTranscript,
    onUserFinalTranscript,
    transcriptions,
  ])

  useEffect(() => {
    onAgentStateChange(agentState)
  }, [agentState, onAgentStateChange])

  const microphoneState = !microphoneTrack
    ? 'Starting…'
    : microphoneTrack.isMuted
      ? 'Muted'
      : 'Listening'

  return (
    <div className="voice-room">
      <div className="voice-device-state" role="status">
        <span>Microphone: {microphoneState}</span>
        <span>Transport agent: {formatState(agentState)}</span>
      </div>
      <VoiceAssistantControlBar
        controls={{ leave: false, microphone: true }}
        onDeviceError={({ error }) => onError(error.message)}
      />
      <StartAudio label="Enable speaker audio" className="start-audio-button" />
      <RoomAudioRenderer />
    </div>
  )
}

function App() {
  const [email, setEmail] = useState('')
  const [pin, setPin] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [customerId, setCustomerId] = useState<string | null>(null)
  const [authenticated, setAuthenticated] = useState(false)
  const [conversationPhase, setConversationPhase] = useState('NOT_STARTED')
  const [turnStatus, setTurnStatus] = useState('READY')
  const [pendingAction, setPendingAction] = useState<string | null>(null)
  const [executedTools, setExecutedTools] = useState<string[]>([])
  const [messages, setMessages] = useState<TranscriptMessage[]>([])
  const [draft, setDraft] = useState('')
  const [authenticating, setAuthenticating] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [voiceCredentials, setVoiceCredentials] =
    useState<VoiceConnectionToken | null>(null)
  const [voiceConnectionState, setVoiceConnectionState] =
    useState<VoiceConnectionState>('disconnected')
  const [voiceAgentState, setVoiceAgentState] = useState('idle')
  const [voiceTurnPending, setVoiceTurnPending] = useState(false)
  const [voiceInterrupted, setVoiceInterrupted] = useState(false)
  const [voiceError, setVoiceError] = useState<string | null>(null)
  const [sessionStarting, setSessionStarting] = useState(true)
  const [resetting, setResetting] = useState(false)
  const transcriptRef = useRef<HTMLDivElement>(null)
  const transcriptEndRef = useRef<HTMLDivElement>(null)
  const composerRef = useRef<HTMLTextAreaElement>(null)
  const shouldAutoScroll = useRef(true)
  const latestVoiceAssistantId = useRef<string | null>(null)
  const pendingVoiceInterruptions = useRef(0)

  const updateSessionState = useCallback((response: SessionResponse) => {
    setSessionId(response.session_id)
    setCustomerId(response.customer_id)
    setAuthenticated(response.authenticated)
    setConversationPhase(response.conversation_phase)
    if (response.turn_status) {
      setTurnStatus(response.turn_status)
    }
    setPendingAction(response.pending_action)
  }, [])

  useEffect(() => {
    let ignore = false

    void createInitialSession()
      .then((session) => {
        if (!ignore) {
          updateSessionState(session)
        }
      })
      .catch((requestError: unknown) => {
        if (!ignore) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : 'An anonymous session could not be created.',
          )
        }
      })
      .finally(() => {
        if (!ignore) {
          setSessionStarting(false)
        }
      })

    return () => {
      ignore = true
    }
  }, [updateSessionState])

  const refreshSessionState = async (attachSources = false) => {
    if (!sessionId) {
      return
    }

    try {
      const response = await getJson<SessionResponse>(`/sessions/${sessionId}`)
      updateSessionState(response)
      if (attachSources && response.policy_sources.length > 0) {
        setMessages((current) => {
          const index = current.findLastIndex(
            (message) => message.role === 'assistant',
          )
          if (index < 0) {
            return current
          }
          return current.map((message, messageIndex) =>
            messageIndex === index
              ? { ...message, sources: response.policy_sources }
              : message,
          )
        })
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Could not refresh the authoritative session state.',
      )
    }
  }

  const addVoiceTranscript = (message: TranscriptMessage) => {
    setMessages((current) => {
      if (current.some((existing) => existing.id === message.id)) {
        return current
      }

      if (message.role === 'assistant') {
        if (pendingVoiceInterruptions.current > 0) {
          pendingVoiceInterruptions.current -= 1
          return [...current, { ...message, interrupted: true }]
        }
        latestVoiceAssistantId.current = message.id
      }

      return [...current, message]
    })
  }

  const handleVoiceInterruption = (event: VoiceInterruptionEvent) => {
    const messageId = latestVoiceAssistantId.current
    latestVoiceAssistantId.current = null

    if (messageId) {
      setMessages((current) =>
        current.map((message) =>
          message.id === messageId
            ? { ...message, interrupted: true }
            : message,
        ),
      )
    } else {
      pendingVoiceInterruptions.current += 1
    }

    setVoiceInterrupted(true)
    setVoiceTurnPending(false)
    setConversationPhase('INTERRUPTED')
    void refreshSessionState()
    console.info('Voice playback interrupted', {
      speechId: event.speech_id,
      stopLatencyMs: event.interruption_stop_latency_ms,
    })
  }

  useEffect(() => {
    const composer = composerRef.current

    if (!composer) {
      return
    }

    composer.style.height = 'auto'
    composer.style.height = `${Math.min(composer.scrollHeight, 112)}px`
  }, [draft])

  useEffect(() => {
    if (shouldAutoScroll.current) {
      transcriptEndRef.current?.scrollIntoView({ block: 'end' })
    }
  }, [messages, processing])

  const handleTranscriptScroll = () => {
    const transcript = transcriptRef.current

    if (!transcript) {
      return
    }

    const distanceFromBottom =
      transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight
    shouldAutoScroll.current = distanceFromBottom < 72
  }

  const startVoice = async () => {
    if (
      !canStartVoice({
        sessionId,
        authenticated,
        voiceConnectionState,
      })
    ) {
      return
    }

    setError(null)
    setVoiceError(null)
    setVoiceTurnPending(false)
    setVoiceInterrupted(false)
    setVoiceAgentState('idle')
    setVoiceConnectionState('connecting')
    latestVoiceAssistantId.current = null
    pendingVoiceInterruptions.current = 0

    try {
      const credentials = await postJson<VoiceConnectionToken>(
        `/sessions/${sessionId}/voice/token`,
      )
      setVoiceCredentials(credentials)
    } catch (requestError) {
      setVoiceConnectionState('disconnected')
      const message =
        requestError instanceof Error
          ? requestError.message
          : 'Voice could not be started.'
      setVoiceError(message)
      setError(message)
    }
  }

  const endVoice = () => {
    setVoiceCredentials(null)
    setVoiceConnectionState('disconnected')
    setVoiceAgentState('idle')
    setVoiceTurnPending(false)
    setVoiceInterrupted(false)
    setVoiceError(null)
    latestVoiceAssistantId.current = null
    pendingVoiceInterruptions.current = 0
  }

  const resetDemoSession = async () => {
    setResetting(true)
    endVoice()
    setSessionId(null)
    setCustomerId(null)
    setAuthenticated(false)
    setConversationPhase('NOT_STARTED')
    setTurnStatus('READY')
    setPendingAction(null)
    setExecutedTools([])
    setMessages([])
    setDraft('')
    setEmail('')
    setPin('')
    setError(null)
    shouldAutoScroll.current = true

    try {
      const session = await postJson<SessionResponse>('/sessions')
      updateSessionState(session)
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'A new demo session could not be created.',
      )
    } finally {
      setResetting(false)
    }
  }

  const startAndSignIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setAuthenticating(true)
    setError(null)
    setVoiceError(null)

    try {
      let activeSessionId = sessionId

      if (!activeSessionId) {
        const session = await postJson<SessionResponse>('/sessions')
        activeSessionId = session.session_id
        updateSessionState(session)
      }

      const authenticatedSession = await postJson<SessionResponse>(
        `/sessions/${activeSessionId}/authenticate`,
        { email, pin },
      )

      updateSessionState(authenticatedSession)
      setTurnStatus('READY')
      setMessages((current) =>
        current.length > 0
          ? current
          : [
              newTranscriptMessage(
                'assistant',
                'You’re signed in. How can I help with your banking today?',
              ),
            ],
      )
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Sign-in failed.',
      )
    } finally {
      setPin('')
      setAuthenticating(false)
    }
  }

  const sendMessage = async (text: string) => {
    const normalized = text.trim()

    if (!normalized || !sessionId || processing) {
      return
    }

    setMessages((current) => [
      ...current,
      newTranscriptMessage('user', normalized),
    ])
    setProcessing(true)
    shouldAutoScroll.current = true
    setError(null)
    setTurnStatus('PROCESSING')

    try {
      const response = await postJson<MessageResponse>(
        `/sessions/${sessionId}/messages`,
        { message: normalized },
      )

      updateSessionState(response)
      setTurnStatus(response.turn_status)
      setExecutedTools(response.executed_tools)
      setMessages((current) => [
        ...current,
        newTranscriptMessage(
          'assistant',
          response.message,
          response.policy_sources,
        ),
      ])
    } catch (requestError) {
      setTurnStatus('FAILED')
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'SentinelVoice could not complete that turn.',
      )
    } finally {
      setProcessing(false)
    }
  }

  const submitMessage = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    submitDraft()
  }

  const submitDraft = () => {
    if (!draft.trim() || !textChatEnabled) {
      return
    }

    const message = draft
    setDraft('')
    void sendMessage(message)
  }

  const handleComposerKeyDown = (
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault()
      submitDraft()
    }
  }

  const waitingForConfirmation =
    turnStatus === 'WAITING_FOR_CONFIRMATION'
  const textChatEnabled = canUseTextChat({ sessionId, processing })
  const voiceStartEnabled = canStartVoice({
    sessionId,
    authenticated,
    voiceConnectionState,
  })
  const voiceStatus = voiceError
    ? 'Error'
    : voiceConnectionState === 'connecting'
      ? 'Connecting'
      : voiceConnectionState === 'disconnected'
        ? 'Disconnected'
        : voiceInterrupted
          ? 'Interrupted'
        : voiceAgentState === 'speaking'
          ? 'Speaking'
          : voiceTurnPending || voiceAgentState === 'thinking'
            ? 'Processing'
            : 'Listening'
  const conversationStatus = processing
    ? 'Processing'
    : voiceConnectionState === 'connected'
      ? voiceStatus
      : 'Ready'

  return (
    <main className="app-shell">
      <header className="brand-header">
        <div className="brand-mark" aria-hidden="true">
          S
        </div>
        <div>
          <p className="eyebrow">Synthetic Digital Bank Support</p>
          <h1>SentinelVoice</h1>
        </div>
        <div className="header-actions">
          <span className="mode-pill">Voice + text test console</span>
          <button
            type="button"
            className="reset-button"
            onClick={() => void resetDemoSession()}
            disabled={
              sessionStarting || resetting || authenticating || processing
            }
          >
            {resetting ? 'Resetting…' : 'New Session'}
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <section className="panel auth-panel" aria-labelledby="auth-heading">
            <div className="panel-heading">
              <div>
                <p className="section-label">Secure access</p>
                <h2 id="auth-heading">Demo sign in</h2>
              </div>
              <span
                className={`status-dot ${authenticated ? 'online' : ''}`}
                aria-hidden="true"
              />
            </div>

            <form onSubmit={startAndSignIn} className="auth-form">
              <label>
                Synthetic customer email
                <input
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="name@example.test"
                  autoComplete="username"
                  required
                  disabled={authenticating || authenticated}
                />
              </label>
              <label>
                Demo PIN
                <input
                  type="password"
                  value={pin}
                  onChange={(event) => setPin(event.target.value)}
                  placeholder="Enter configured PIN"
                  autoComplete="current-password"
                  required
                  disabled={authenticating || authenticated}
                />
              </label>
              <button
                className="primary-button"
                type="submit"
                disabled={sessionStarting || authenticating || authenticated}
              >
                {authenticated
                  ? 'Signed in'
                  : sessionStarting || authenticating
                    ? 'Starting session…'
                    : 'Start / Sign In'}
              </button>
            </form>

            <p className="auth-status" role="status">
              {authenticated
                ? 'Authenticated synthetic customer session'
                : sessionId
                  ? 'Session started · Sign-in required for private banking tools'
                  : 'No active session'}
            </p>
          </section>

          <section className="panel status-panel" aria-labelledby="status-heading">
            <div className="panel-heading">
              <div>
                <p className="section-label">Server state</p>
                <h2 id="status-heading">Session details</h2>
              </div>
            </div>

            <dl className="state-list">
              <div>
                <dt>Authentication</dt>
                <dd>{authenticated ? 'Authenticated' : 'Unauthenticated'}</dd>
              </div>
              <div>
                <dt>Conversation phase</dt>
                <dd>{formatState(conversationPhase)}</dd>
              </div>
              <div>
                <dt>Turn status</dt>
                <dd>{formatState(turnStatus)}</dd>
              </div>
              <div>
                <dt>Pending action</dt>
                <dd>{pendingAction ? formatState(pendingAction) : 'None'}</dd>
              </div>
            </dl>

            <div className="identifier-block">
              <span>Session ID</span>
              <code>{sessionId ?? 'Not created'}</code>
            </div>
            {customerId && (
              <div className="identifier-block">
                <span>Verified customer ID</span>
                <code>{customerId}</code>
              </div>
            )}

            <div className="tool-activity">
              <span>Latest tool activity</span>
              {executedTools.length > 0 ? (
                <ul>
                  {executedTools.map((tool) => (
                    <li key={tool}>{formatState(tool)}</li>
                  ))}
                </ul>
              ) : (
                <p>No tools executed in the latest turn.</p>
              )}
            </div>
          </section>
          <section className="panel voice-panel" aria-labelledby="voice-heading">
            <div className="panel-heading">
              <div>
                <p className="section-label">WebRTC voice</p>
                <h2 id="voice-heading">Voice session</h2>
              </div>
            </div>

            <div className={`voice-summary ${voiceStatus.toLowerCase()}`}>
              <span className="voice-dot" aria-hidden="true" />
              <div>
                <strong>{voiceStatus}</strong>
                <span>
                  {voiceError ??
                    (authenticated
                      ? 'Microphone and agent transport'
                      : 'Sign in to enable voice')}
                </span>
              </div>
            </div>

            {voiceCredentials ? (
              <LiveKitRoom
                token={voiceCredentials.participant_token}
                serverUrl={voiceCredentials.server_url}
                connect
                audio
                video={false}
                onConnected={() => {
                  setVoiceConnectionState('connected')
                  setVoiceError(null)
                }}
                onDisconnected={() => {
                  setVoiceCredentials(null)
                  setVoiceConnectionState('disconnected')
                  setVoiceAgentState('idle')
                  setVoiceTurnPending(false)
                  setVoiceInterrupted(false)
                  latestVoiceAssistantId.current = null
                  pendingVoiceInterruptions.current = 0
                }}
                onError={(voiceError) => {
                  const message = `Voice connection failed: ${voiceError.message}`
                  setError(message)
                  setVoiceError(message)
                  setVoiceCredentials(null)
                  setVoiceConnectionState('disconnected')
                  latestVoiceAssistantId.current = null
                  pendingVoiceInterruptions.current = 0
                }}
                onMediaDeviceFailure={() => {
                  const message =
                    'Microphone access failed. Allow microphone permission and try again.'
                  setError(message)
                  setVoiceError(message)
                }}
              >
                <VoiceRoom
                  onAgentStateChange={(state) => {
                    setVoiceAgentState(state)
                    if (state === 'listening' || state === 'idle') {
                      latestVoiceAssistantId.current = null
                    }
                  }}
                  onAgentFinalTranscript={() => {
                    setVoiceTurnPending(false)
                    setVoiceInterrupted(false)
                    void refreshSessionState(true)
                  }}
                  onError={(message) => {
                    setVoiceError(message)
                    setError(message)
                    setVoiceTurnPending(false)
                  }}
                  onTranscript={addVoiceTranscript}
                  onInterruption={handleVoiceInterruption}
                  onUserFinalTranscript={() => {
                    setVoiceError(null)
                    setVoiceInterrupted(false)
                    setVoiceTurnPending(true)
                    setTurnStatus('PROCESSING')
                  }}
                />
              </LiveKitRoom>
            ) : (
              <p className="voice-status" role="status">
                {authenticated
                  ? 'Ready to connect microphone audio securely.'
                  : 'Sign in before starting voice.'}
              </p>
            )}

            <div className="voice-actions">
              <button
                type="button"
                className="primary-button"
                onClick={() => void startVoice()}
                disabled={!voiceStartEnabled}
              >
                {voiceConnectionState === 'connecting'
                  ? 'Connecting…'
                  : 'Start Voice'}
              </button>
              <button
                type="button"
                className="end-voice-button"
                onClick={endVoice}
                disabled={!voiceCredentials}
              >
                End Voice
              </button>
            </div>
          </section>
        </aside>

        <section className="conversation-panel" aria-labelledby="conversation-heading">
          <div className="conversation-top">
            <div className="conversation-header">
              <div>
                <p className="section-label">Conversation</p>
                <h2 id="conversation-heading">Banking support</h2>
              </div>
              <div
                className={`agent-state ${conversationStatus.toLowerCase()}`}
                role="status"
              >
                <span aria-hidden="true" />
                {conversationStatus}
              </div>
            </div>

            {error && (
              <div className="error-banner" role="alert">
                {error}
              </div>
            )}
          </div>

          <div
            className="transcript"
            aria-live="polite"
            ref={transcriptRef}
            onScroll={handleTranscriptScroll}
          >
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon" aria-hidden="true">
                  ◌
                </div>
                <h3>Start a secure demo session</h3>
                <p>
                  Sign in with a synthetic customer email and your locally
                  configured demo PIN, then ask a banking-support question.
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <article
                  key={message.id}
                  className={`message ${message.role}${
                    message.interrupted ? ' interrupted' : ''
                  }`}
                >
                  <p className="message-author">
                    {message.role === 'assistant' ? 'SentinelVoice' : 'You'}
                    {message.interrupted && (
                      <span className="message-interruption">
                        Speech interrupted
                      </span>
                    )}
                  </p>
                  <p className="message-body">{message.text}</p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources" aria-label="Policy sources">
                      <span>Sources</span>
                      <ul>
                        {message.sources.map((source) => (
                          <li key={source}>{source}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </article>
              ))
            )}

            {processing && (
              <div className="typing-indicator" aria-label="SentinelVoice is processing">
                <span />
                <span />
                <span />
              </div>
            )}
            <div ref={transcriptEndRef} aria-hidden="true" />
          </div>

          <div className="conversation-footer">
            {waitingForConfirmation && (
              <div className="confirmation-card" role="status">
                <div>
                  <p className="confirmation-title">Confirmation required</p>
                  <p>
                    The protected action is still pending on the server. Choose
                    whether SentinelVoice should continue.
                  </p>
                </div>
                <div className="confirmation-actions">
                  <button
                    type="button"
                    className="confirm-button"
                    onClick={() => void sendMessage('Yes')}
                    disabled={processing}
                  >
                    Confirm
                  </button>
                  <button
                    type="button"
                    className="cancel-button"
                    onClick={() => void sendMessage('Cancel')}
                    disabled={processing}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            <form className="composer" onSubmit={submitMessage}>
              <label className="sr-only" htmlFor="message-input">
                Message SentinelVoice
              </label>
              <textarea
                id="message-input"
                ref={composerRef}
                rows={1}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder={
                  sessionStarting
                    ? 'Starting a secure session…'
                    : authenticated
                      ? 'Ask about your synthetic bank account…'
                      : 'Ask a public policy question or sign in for account help.'
                }
                maxLength={4000}
                disabled={!textChatEnabled}
              />
              <button
                type="submit"
                className="send-button"
                disabled={!textChatEnabled || !draft.trim()}
              >
                Send
                <span aria-hidden="true">↗</span>
              </button>
            </form>
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
