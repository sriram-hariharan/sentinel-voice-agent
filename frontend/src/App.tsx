import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type SessionResponse = {
  session_id: string
  customer_id: string | null
  authenticated: boolean
  conversation_phase: string
  pending_action: string | null
}

type MessageResponse = SessionResponse & {
  message: string
  turn_status: string
  executed_tools: string[]
}

type TranscriptMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
}

function newTranscriptMessage(
  role: TranscriptMessage['role'],
  text: string,
): TranscriptMessage {
  return { id: crypto.randomUUID(), role, text }
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

function formatState(value: string): string {
  return value.toLowerCase().replaceAll('_', ' ')
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

  const updateSessionState = (response: SessionResponse) => {
    setSessionId(response.session_id)
    setCustomerId(response.customer_id)
    setAuthenticated(response.authenticated)
    setConversationPhase(response.conversation_phase)
    setPendingAction(response.pending_action)
  }

  const startAndSignIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setAuthenticating(true)
    setError(null)

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
        newTranscriptMessage('assistant', response.message),
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
    const message = draft
    setDraft('')
    void sendMessage(message)
  }

  const waitingForConfirmation =
    turnStatus === 'WAITING_FOR_CONFIRMATION'

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
        <span className="mode-pill">Text preview</span>
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
                disabled={authenticating || authenticated}
              >
                {authenticated
                  ? 'Signed in'
                  : authenticating
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
        </aside>

        <section className="conversation-panel" aria-labelledby="conversation-heading">
          <div className="conversation-header">
            <div>
              <p className="section-label">Conversation</p>
              <h2 id="conversation-heading">Banking support</h2>
            </div>
            <div className={`agent-state ${processing ? 'busy' : ''}`}>
              <span aria-hidden="true" />
              {processing ? 'Processing' : 'Ready'}
            </div>
          </div>

          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}

          <div className="transcript" aria-live="polite">
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
                  className={`message ${message.role}`}
                >
                  <p className="message-author">
                    {message.role === 'assistant' ? 'SentinelVoice' : 'You'}
                  </p>
                  <p>{message.text}</p>
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
          </div>

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
            <input
              id="message-input"
              type="text"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={
                sessionId
                  ? 'Ask about your synthetic bank account…'
                  : 'Start a session to send a message'
              }
              maxLength={4000}
              disabled={!sessionId || processing}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!sessionId || processing || !draft.trim()}
            >
              Send
              <span aria-hidden="true">↗</span>
            </button>
          </form>
        </section>
      </section>
    </main>
  )
}

export default App
