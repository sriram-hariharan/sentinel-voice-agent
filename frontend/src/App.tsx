import {
  LiveKitRoom,
  RoomAudioRenderer,
  StartAudio,
  useDataChannel,
  useIsSpeaking,
  useLocalParticipant,
  useTranscriptions,
  useVoiceAssistant,
} from '@livekit/components-react'
import '@livekit/components-styles'
import { Plus, SendHorizontal } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import './App.css'
import {
  canStartVoice,
  canUseTextChat,
  type VoiceConnectionState,
} from './access'
import { SessionControlBar } from './components/SessionControlBar'
import { TraceMetricsPanel } from './TraceMetricsPanel'
import type { SessionObservabilityResponse } from './traceMetrics'
import {
  deriveVoiceUiState,
  requestMicrophoneToggle,
  voiceUiStateLabel,
  type MicrophoneController,
} from './voiceActivity'

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

type VoiceRoomProps = {
  onAgentStateChange: (state: string) => void
  onAgentFinalTranscript: () => void
  onError: (message: string) => void
  onInterruption: (event: VoiceInterruptionEvent) => void
  onTranscript: (message: TranscriptMessage) => void
  onUserFinalTranscript: () => void
  onUserSpeakingChange: (speaking: boolean) => void
  onMicrophoneControllerChange: (
    controller: MicrophoneController | null,
  ) => void
  onMicrophoneEnabledChange: (enabled: boolean) => void
}

function VoiceRoom({
  onAgentStateChange,
  onAgentFinalTranscript,
  onError,
  onInterruption,
  onTranscript,
  onUserFinalTranscript,
  onUserSpeakingChange,
  onMicrophoneControllerChange,
  onMicrophoneEnabledChange,
}: VoiceRoomProps) {
  const transcriptions = useTranscriptions()
  const { isMicrophoneEnabled, localParticipant } = useLocalParticipant()
  const userSpeaking = useIsSpeaking(localParticipant)
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

  useEffect(() => {
    onUserSpeakingChange(isMicrophoneEnabled && userSpeaking)
  }, [isMicrophoneEnabled, onUserSpeakingChange, userSpeaking])

  useEffect(() => {
    onMicrophoneEnabledChange(isMicrophoneEnabled)
  }, [isMicrophoneEnabled, onMicrophoneEnabledChange])

  const setMicrophoneEnabled = useCallback<MicrophoneController>(
    async (enabled) => {
      await localParticipant.setMicrophoneEnabled(enabled)
      return localParticipant.isMicrophoneEnabled
    },
    [localParticipant],
  )

  useEffect(() => {
    onMicrophoneControllerChange(setMicrophoneEnabled)
    return () => onMicrophoneControllerChange(null)
  }, [onMicrophoneControllerChange, setMicrophoneEnabled])

  useEffect(
    () => () => onUserSpeakingChange(false),
    [onUserSpeakingChange],
  )

  return (
    <div className="voice-room" aria-label="Connected voice controls">
      <StartAudio label="Enable speaker audio" className="start-audio-button" />
      <RoomAudioRenderer />
    </div>
  )
}

function App() {
  const [email, setEmail] = useState('')
  const [pin, setPin] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [authenticated, setAuthenticated] = useState(false)
  const [conversationPhase, setConversationPhase] = useState('NOT_STARTED')
  const [turnStatus, setTurnStatus] = useState('READY')
  const [pendingAction, setPendingAction] = useState<string | null>(null)
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
  const [userSpeaking, setUserSpeaking] = useState(false)
  const [microphoneEnabled, setMicrophoneEnabled] = useState(true)
  const [microphoneControllerReady, setMicrophoneControllerReady] =
    useState(false)
  const [microphoneTogglePending, setMicrophoneTogglePending] = useState(false)
  const [voiceTurnPending, setVoiceTurnPending] = useState(false)
  const [voiceInterrupted, setVoiceInterrupted] = useState(false)
  const [voiceError, setVoiceError] = useState<string | null>(null)
  const [sessionStarting, setSessionStarting] = useState(true)
  const [resetting, setResetting] = useState(false)
  const [observability, setObservability] =
    useState<SessionObservabilityResponse | null>(null)
  const [observabilityLoading, setObservabilityLoading] = useState(false)
  const [observabilityError, setObservabilityError] = useState<string | null>(
    null,
  )
  const transcriptRef = useRef<HTMLDivElement>(null)
  const transcriptEndRef = useRef<HTMLDivElement>(null)
  const composerRef = useRef<HTMLTextAreaElement>(null)
  const shouldAutoScroll = useRef(true)
  const latestVoiceAssistantId = useRef<string | null>(null)
  const pendingVoiceInterruptions = useRef(0)
  const observabilityRequestId = useRef(0)
  const microphoneControllerRef = useRef<MicrophoneController | null>(null)

  const handleMicrophoneControllerChange = useCallback(
    (controller: MicrophoneController | null) => {
      microphoneControllerRef.current = controller
      setMicrophoneControllerReady(controller !== null)
    },
    [],
  )

  const updateSessionState = useCallback((response: SessionResponse) => {
    setSessionId(response.session_id)
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

  const refreshObservability = useCallback(
    async (targetSessionId: string | null = sessionId) => {
      const requestId = observabilityRequestId.current + 1
      observabilityRequestId.current = requestId

      if (!targetSessionId) {
        setObservability(null)
        setObservabilityError(null)
        setObservabilityLoading(false)
        return
      }

      setObservabilityLoading(true)
      setObservabilityError(null)

      try {
        const response = await getJson<SessionObservabilityResponse>(
          `/sessions/${targetSessionId}/observability`,
        )
        if (requestId === observabilityRequestId.current) {
          setObservability(response)
        }
      } catch (requestError) {
        if (requestId === observabilityRequestId.current) {
          setObservabilityError(
            requestError instanceof Error
              ? requestError.message
              : 'Trace and metrics could not be refreshed.',
          )
        }
      } finally {
        if (requestId === observabilityRequestId.current) {
          setObservabilityLoading(false)
        }
      }
    },
    [sessionId],
  )

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
    void refreshSessionState().then(() => refreshObservability(sessionId))
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
      const transcript = transcriptRef.current
      if (transcript) {
        transcript.scrollTop = transcript.scrollHeight
      }
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
    setUserSpeaking(false)
    setMicrophoneEnabled(true)
    setMicrophoneTogglePending(false)
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
    microphoneControllerRef.current = null
    setVoiceCredentials(null)
    setVoiceConnectionState('disconnected')
    setVoiceAgentState('idle')
    setUserSpeaking(false)
    setMicrophoneEnabled(true)
    setMicrophoneControllerReady(false)
    setMicrophoneTogglePending(false)
    setVoiceTurnPending(false)
    setVoiceInterrupted(false)
    setVoiceError(null)
    latestVoiceAssistantId.current = null
    pendingVoiceInterruptions.current = 0
  }

  const toggleMicrophone = async () => {
    if (microphoneTogglePending) {
      return
    }

    setMicrophoneTogglePending(true)

    try {
      const authoritativeEnabled = await requestMicrophoneToggle({
        connected: voiceConnectionState === 'connected',
        microphoneEnabled,
        controller: microphoneControllerRef.current,
      })

      if (authoritativeEnabled !== null) {
        setMicrophoneEnabled(authoritativeEnabled)
        setUserSpeaking(false)
        setVoiceError(null)
      }
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? `Microphone update failed: ${requestError.message}`
          : 'Microphone update failed. Try again.'
      setVoiceError(message)
      setError(message)
    } finally {
      setMicrophoneTogglePending(false)
    }
  }

  const resetDemoSession = async () => {
    setResetting(true)
    endVoice()
    setSessionId(null)
    setAuthenticated(false)
    setConversationPhase('NOT_STARTED')
    setTurnStatus('READY')
    setPendingAction(null)
    setMessages([])
    setDraft('')
    setEmail('')
    setPin('')
    setError(null)
    observabilityRequestId.current += 1
    setObservability(null)
    setObservabilityError(null)
    setObservabilityLoading(false)
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
      setMessages((current) => [
        ...current,
        newTranscriptMessage(
          'assistant',
          response.message,
          response.policy_sources,
        ),
      ])
      void refreshObservability(sessionId)
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
  const voiceUiState = deriveVoiceUiState({
    voiceConnectionState,
    voiceAgentState,
    microphoneMuted:
      voiceConnectionState === 'connected' && !microphoneEnabled,
    userSpeaking: microphoneEnabled && userSpeaking,
    voiceTurnPending,
    voiceInterrupted,
    voiceError,
  })
  const voiceStatus = voiceUiStateLabel(voiceUiState)
  const conversationStatus = processing
    ? 'Processing'
    : voiceConnectionState === 'connected'
      ? voiceStatus
      : 'Ready'

  const voiceControls = voiceCredentials ? (
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
        microphoneControllerRef.current = null
        setVoiceCredentials(null)
        setVoiceConnectionState('disconnected')
        setVoiceAgentState('idle')
        setUserSpeaking(false)
        setMicrophoneEnabled(true)
        setMicrophoneControllerReady(false)
        setMicrophoneTogglePending(false)
        setVoiceTurnPending(false)
        setVoiceInterrupted(false)
        latestVoiceAssistantId.current = null
        pendingVoiceInterruptions.current = 0
      }}
      onError={(connectionError) => {
        const message = `Voice connection failed: ${connectionError.message}`
        setError(message)
        setVoiceError(message)
        setVoiceCredentials(null)
        setVoiceConnectionState('disconnected')
        setUserSpeaking(false)
        setMicrophoneEnabled(true)
        setMicrophoneControllerReady(false)
        setMicrophoneTogglePending(false)
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
          void refreshSessionState(true).then(() =>
            refreshObservability(sessionId),
          )
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
        onUserSpeakingChange={setUserSpeaking}
        onMicrophoneControllerChange={handleMicrophoneControllerChange}
        onMicrophoneEnabledChange={setMicrophoneEnabled}
      />
    </LiveKitRoom>
  ) : null

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
            <Plus size={16} aria-hidden="true" />
            {resetting ? 'Resetting…' : 'New Session'}
          </button>
        </div>
      </header>

      <SessionControlBar
        authenticated={authenticated}
        authenticating={authenticating}
        email={email}
        pin={pin}
        sessionId={sessionId}
        sessionStarting={sessionStarting}
        conversationPhase={conversationPhase}
        turnStatus={turnStatus}
        pendingAction={pendingAction}
        voiceUiState={voiceUiState}
        voiceError={voiceError}
        voiceStartEnabled={voiceStartEnabled}
        voiceActive={voiceCredentials !== null}
        microphoneEnabled={microphoneEnabled}
        microphoneToggleEnabled={
          voiceConnectionState === 'connected' && microphoneControllerReady
        }
        microphoneTogglePending={microphoneTogglePending}
        onEmailChange={setEmail}
        onPinChange={setPin}
        onSignIn={startAndSignIn}
        onStartVoice={() => void startVoice()}
        onEndVoice={endVoice}
        onToggleMicrophone={() => void toggleMicrophone()}
        voiceControls={voiceControls}
      />

      <section className="workspace">
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
                  {message.role === 'assistant' && (
                    <div className="message-avatar" aria-hidden="true">
                      S
                    </div>
                  )}
                  <div className="message-content">
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
                  </div>
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
                <SendHorizontal size={15} aria-hidden="true" />
              </button>
            </form>
          </div>
        </section>

        <TraceMetricsPanel
          response={observability}
          loading={observabilityLoading}
          error={observabilityError}
          canRefresh={sessionId !== null}
          onRefresh={() => void refreshObservability()}
        />
      </section>
    </main>
  )
}

export default App
