
import { useRef, useState } from "react";
import "./App.css";

const API = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000/api/v1"
).replace(/\/$/, "");

type Source = Record<string, unknown>;

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  verification?: Record<string, unknown>;
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [showSources, setShowSources] = useState<number | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  async function uploadDocument() {
    if (!file || uploading) return;

    setUploading(true);
    setError("");
    setNotice("");

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API}/documents/upload`, {
        method: "POST",
        body: form,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document upload failed.");
      }

      setNotice(
        `Successfully indexed ${data.filename}. ${
          data.chunks_created ?? 0
        } chunks created.`
      );
      setFile(null);

      if (fileInput.current) fileInput.current.value = "";
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not connect to the backend."
      );
    } finally {
      setUploading(false);
    }
  }

  async function sendQuestion(text = question) {
    const q = text.trim();
    if (!q || sending) return;

    setMessages((old) => [...old, { role: "user", content: q }]);
    setQuestion("");
    setSending(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not generate an answer.");
      }

      setMessages((old) => [
        ...old,
        {
          role: "assistant",
          content: data.answer ?? "No answer was returned.",
          sources: Array.isArray(data.sources) ? data.sources : [],
          verification: data.verification ?? {},
        },
      ]);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not connect to the backend."
      );
    } finally {
      setSending(false);
    }
  }

  function clearConversation() {
    setMessages([]);
    setError("");
    setNotice("");
    setShowSources(null);
  }

  function sourceText(source: Source) {
    return String(
      source.source ??
        source.filename ??
        source.file_name ??
        source.document ??
        source.title ??
        "Document source"
    );
  }

  function sourceDetails(source: Source) {
    return Object.entries(source)
      .filter(([, value]) => value !== null && value !== undefined)
      .map(([key, value]) => `${key}: ${
        typeof value === "object" ? JSON.stringify(value) : String(value)
      }`)
      .join("\n");
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">R</div>
          <div>
            <strong>RAG Studio</strong>
            <span>Knowledge assistant</span>
          </div>
        </div>

        <button className="new-chat" onClick={clearConversation}>
          <span>＋</span> New conversation
        </button>

        <div className="side-label">WORKSPACE</div>
        <div className="side-item active">▤ &nbsp; Document assistant</div>
        <div className="side-item">⌕ &nbsp; Hybrid retrieval</div>
        <div className="side-item">✓ &nbsp; Answer verification</div>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <div>
            <strong>Local API</strong>
            <span>Expected at localhost:8000</span>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">YOUR KNOWLEDGE BASE</div>
            <h1>Hybrid RAG Assistant</h1>
          </div>
          <span className="model-pill">Groq · RAG</span>
        </header>

        <div className="content">
          <section className="upload-card">
            <div className="upload-heading">
              <div className="upload-icon">↑</div>
              <div>
                <h2>Upload your documents</h2>
                <p>Give your assistant knowledge to search.</p>
              </div>
            </div>

            <div className="upload-controls">
              <input
                ref={fileInput}
                type="file"
                accept=".pdf,.txt,.md,.html,.htm"
                onChange={(e) => {
                  setFile(e.target.files?.[0] ?? null);
                  setNotice("");
                  setError("");
                }}
              />
              <button
                className="primary-button"
                onClick={uploadDocument}
                disabled={!file || uploading}
              >
                {uploading ? "Indexing..." : "Upload & index"}
              </button>
            </div>

            <p className="hint">
              PDF, TXT, Markdown or HTML · Maximum 20 MB
            </p>
            {notice && <div className="notice">{notice}</div>}
            {error && <div className="error">{error}</div>}
          </section>

          <section className="chat-panel">
            <div className="chat-heading">
              <div>
                <h2>Ask your documents</h2>
                <p>Answers are generated using retrieved context.</p>
              </div>
              <span className="online-label">● Ready</span>
            </div>

            <div className="messages">
              {messages.length === 0 && (
                <div className="welcome">
                  <div className="welcome-symbol">✳</div>
                  <h2>What would you like to know?</h2>
                  <p>
                    Upload a document, then ask a question. The assistant
                    will retrieve relevant passages and generate an answer.
                  </p>
                  <div className="suggestions">
                    {[
                      "Summarize the main points",
                      "What are the key topics?",
                      "Explain the most important findings",
                    ].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => sendQuestion(suggestion)}
                        disabled={sending}
                      >
                        {suggestion} <span>↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((message, index) => (
                <article
                  className={`message ${message.role}`}
                  key={`${index}-${message.role}`}
                >
                  <div className="avatar">
                    {message.role === "user" ? "Y" : "R"}
                  </div>
                  <div className="message-body">
                    <div className="message-label">
                      {message.role === "user" ? "You" : "RAG Assistant"}
                    </div>
                    <div className="message-text">{message.content}</div>

                    {message.role === "assistant" &&
                      ((message.sources?.length ?? 0) > 0 ||
                        Object.keys(message.verification ?? {}).length >
                          0) && (
                        <div className="answer-tools">
                          {(message.sources?.length ?? 0) > 0 && (
                            <button
                              className="tool-button"
                              onClick={() =>
                                setShowSources(
                                  showSources === index ? null : index
                                )
                              }
                            >
                              {showSources === index
                                ? "Hide sources"
                                : `View sources (${message.sources?.length})`}
                            </button>
                          )}

                          {Object.keys(message.verification ?? {}).length >
                            0 && (
                            <details className="verification">
                              <summary>Verification details</summary>
                              <pre>
                                {JSON.stringify(
                                  message.verification,
                                  null,
                                  2
                                )}
                              </pre>
                            </details>
                          )}
                        </div>
                      )}

                    {showSources === index &&
                      message.sources?.map((source, sourceIndex) => (
                        <details className="source-item" key={sourceIndex}>
                          <summary>{sourceText(source)}</summary>
                          <pre>{sourceDetails(source)}</pre>
                        </details>
                      ))}
                  </div>
                </article>
              ))}

              {sending && (
                <div className="loading-message">
                  <span className="spinner" />
                  Searching documents and generating an answer...
                </div>
              )}
            </div>

            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                void sendQuestion();
              }}
            >
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void sendQuestion();
                  }
                }}
                placeholder="Ask anything about your documents..."
                rows={2}
                maxLength={4000}
                disabled={sending}
              />
              <div className="composer-footer">
                <span>Enter to send · Shift + Enter for a new line</span>
                <button
                  className="send-button"
                  type="submit"
                  disabled={!question.trim() || sending}
                >
                  {sending ? "..." : "Send ↑"}
                </button>
              </div>
            </form>
          </section>

          <footer className="footer">
            Hybrid retrieval · Reranking · Grounded answers
          </footer>
        </div>
      </section>
    </main>
  );
}
