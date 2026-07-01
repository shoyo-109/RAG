"use client";

import React, { useState, useRef, useEffect } from "react";
import styles from "./page.module.css";
import EmbeddingSpace, { ChunkNode } from "./EmbeddingSpace";

interface Message {
  id: string;
  sender: "user" | "ai";
  text: string;
  thinkingStages?: string[];
  isThinking?: boolean;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // 3D Visualizer States
  const [nodes, setNodes] = useState<ChunkNode[]>([]);
  const [retrievedChunks, setRetrievedChunks] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<"chat" | "space">("chat");
  const [topK, setTopK] = useState<number>(10);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "initial",
      sender: "ai",
      text: "👋 Welcome! Start by dropping a PDF or TXT file on the panel to the left. Once indexed, we can start discussing it.",
    },
  ]);
  const [input, setInput] = useState<string>("");
  const [isChatting, setIsChatting] = useState<boolean>(false);
  const [dragActive, setDragActive] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle Drag Over
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Handle Drop
  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      await processAndUploadFile(droppedFile);
    }
  };

  // Handle File Input Select
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await processAndUploadFile(e.target.files[0]);
    }
  };

  // Trigger File Browse Dialog
  const triggerBrowse = () => {
    fileInputRef.current?.click();
  };

  // Upload and Index Document via FastAPI
  const processAndUploadFile = async (selectedFile: File) => {
    const ext = selectedFile.name.split(".").pop()?.toLowerCase();
    if (ext !== "pdf" && ext !== "txt") {
      setUploadError("Only PDF and TXT documents are supported.");
      return;
    }

    setUploadError(null);
    setFile(selectedFile);
    setIsUploading(true);
    setNodes([]);
    setRetrievedChunks([]);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      setSessionId(data.session_id);

      // Load 3D coordinates for chunking visualisation
      if (data.projections) {
        setNodes(data.projections);
      }

      // Update chat message indicating success
      setMessages((prev) => [
        ...prev,
        {
          id: `upload-${Date.now()}`,
          sender: "ai",
          text: `📄 **${selectedFile.name}** was successfully indexed into advanced knowledge base! Custom HNSW index ready. Try asking questions.`,
        },
      ]);
    } catch (err: any) {
      setUploadError(err.message || "An error occurred during file upload.");
      setFile(null);
    } finally {
      setIsUploading(false);
    }
  };

  // Send Chat message and handle SSE response stream
  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !sessionId || isChatting) return;

    const userQuery = input;
    setInput("");
    setIsChatting(true);

    const messageId = `msg-${Date.now()}`;
    const userMsg: Message = { id: `user-${Date.now()}`, sender: "user", text: userQuery };
    const aiMsg: Message = {
      id: messageId,
      sender: "ai",
      text: "",
      thinkingStages: [],
      isThinking: true
    };

    setMessages((prev) => [...prev, userMsg, aiMsg]);

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("question", userQuery);
    formData.append("top_k", topK.toString());

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to send message to chat API.");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) return;

      let partialLine = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + chunk).split("\n\n");
        partialLine = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const rawContent = line.slice(6);
            const trimmedContent = rawContent.trim();
            if (!rawContent) continue;

            // Check if the data specifies a thinking stage log
            if (trimmedContent.startsWith("stage:")) {
              const stageMsg = trimmedContent.slice(6);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === messageId
                    ? {
                      ...msg,
                      thinkingStages: [...(msg.thinkingStages || []), stageMsg],
                    }
                    : msg
                )
              );
            }
            // Check if the data contains retrieved chunks for visual highlight
            else if (trimmedContent.startsWith("retrieved_chunks:")) {
              try {
                const chunksList = JSON.parse(trimmedContent.slice(17));
                setRetrievedChunks(chunksList);
              } catch (e) {
                console.error("Failed to parse retrieved chunks list", e);
              }
            }
            // Normal token content (preserve space)
            else {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === messageId
                    ? {
                      ...msg,
                      isThinking: false,
                      text: msg.text + rawContent,
                    }
                    : msg
                )
              );
            }
          }
        }
      }
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? { ...msg, isThinking: false, text: `⚠️ Error: ${err.message}` }
            : msg
        )
      );
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <main className={styles.mainContainer}>
      <header className={styles.topHeader}>
        <div className={styles.headerTitle}>
          {/* <span className={styles.logoBadge}>ADVANCED</span> */}
          <h1>Cognitive RAG Hub</h1>
        </div>
        {/* <p className={styles.headerSubtitle}>Hybrid Search, Semantic Cache, Hallucination Verification & Embedding Spaces</p> */}
      </header>

      {/* Tab Selector */}
      <div className={styles.tabContainer}>
        <button
          onClick={() => setActiveTab("chat")}
          className={`${styles.tabButton} ${activeTab === "chat" ? styles.activeTab : ""}`}
        >
          💬 Conversation Dashboard
        </button>
        <button
          onClick={() => setActiveTab("space")}
          className={`${styles.tabButton} ${activeTab === "space" ? styles.activeTab : ""}`}
        >
          🌐 3D Embedding Space
        </button>
      </div>

      <section className={`${styles.workspace} ${activeTab === "chat" ? styles.workspaceChat : styles.workspaceSpace}`}>
        {activeTab === "chat" && (
          <>
            {/* Panel 1: Upload Injector Dashboard */}
            <div className={styles.leftPanel}>
              <div className={styles.cardHeader}>
                <h3>Cognitive Ingestion</h3>
                <p>Deploy documents to semantic vector space</p>
              </div>

              {!file ? (
                <div
                  className={`${styles.dragArea} ${dragActive ? styles.dragActive : ""}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={triggerBrowse}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    className={styles.hiddenInput}
                    onChange={handleFileChange}
                    accept=".pdf,.txt"
                  />
                  <div className={styles.uploadPrompt}>
                    <div className={styles.pulseIcon}>📥</div>
                    <p className={styles.primaryText}>Drag & drop document</p>
                    <p className={styles.secondaryText}>PDF or TXT up to 10MB</p>
                    <button type="button" className={styles.browseButton}>
                      Browse Files
                    </button>
                  </div>
                </div>
              ) : (
                <div className={styles.fileLoadedContainer} style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                  {/* File Metadata Details */}
                  <div style={{ display: "flex", alignItems: "center", gap: "0.85rem", background: "rgba(255,255,255,0.02)", padding: "10px 14px", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "2rem" }}>📄</div>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <p className={styles.fileName} style={{ margin: 0 }}>{file.name}</p>
                      <p className={styles.fileSize} style={{ margin: "2px 0 0 0" }}>{(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                  </div>

                  {/* Dynamic Pipeline Progress Stepper */}
                  <div className={styles.stepperContainer}>
                    <div className={`${styles.stepItem} ${isUploading ? styles.stepActive : styles.stepDone}`}>
                      <span className={styles.stepIcon}>{isUploading ? "⚡" : "✓"}</span>
                      <span>1. Extracting Document Text</span>
                      <span className={styles.stepBadge}>{isUploading ? "Active" : "Done"}</span>
                    </div>
                    
                    <div className={`${styles.stepItem} ${isUploading ? styles.stepActive : styles.stepDone}`}>
                      <span className={styles.stepIcon}>{isUploading ? "⚙️" : "✓"}</span>
                      <span>2. Semantic Percentile Chunking</span>
                      <span className={styles.stepBadge}>{isUploading ? "Active" : "Done"}</span>
                    </div>

                    <div className={`${styles.stepItem} ${isUploading ? styles.stepActive : styles.stepDone}`}>
                      <span className={styles.stepIcon}>{isUploading ? "🕸️" : "✓"}</span>
                      <span>3. Building Chroma HNSW Graph</span>
                      <span className={styles.stepBadge}>{isUploading ? "Active" : "Done"}</span>
                    </div>

                    <div className={`${styles.stepItem} ${isUploading ? styles.stepActive : styles.stepDone}`}>
                      <span className={styles.stepIcon}>{isUploading ? "🗂️" : "✓"}</span>
                      <span>4. Rebuilding BM25 Lexical Index</span>
                      <span className={styles.stepBadge}>{isUploading ? "Active" : "Done"}</span>
                    </div>
                  </div>

                  {/* Document Space Statistics */}
                  {!isUploading && nodes.length > 0 && (
                    <div className={styles.statGrid}>
                      <div className={styles.statCard}>
                        <div className={styles.statValue}>{nodes.length}</div>
                        <div className={styles.statLabel}>Total Chunks</div>
                      </div>
                      <div className={styles.statCard}>
                        <div className={styles.statValue}>HNSW</div>
                        <div className={styles.statLabel}>Graph Mode</div>
                      </div>
                    </div>
                  )}

                  {/* Interactive RAG Parameter Sliders */}
                  {!isUploading && sessionId && (
                    <div style={{ display: "flex", flexDirection: "column", gap: "1rem", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "1.25rem" }}>
                      <div className={styles.parameterSlider}>
                        <div className={styles.sliderLabel}>
                          <span>Context Chunks (K)</span>
                          <span style={{ color: "var(--accent-primary)", fontWeight: "800" }}>{topK}</span>
                        </div>
                        <input
                          type="range"
                          min="3"
                          max="15"
                          value={topK}
                          onChange={(e) => setTopK(parseInt(e.target.value))}
                          className={styles.sliderInput}
                        />
                      </div>

                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "rgba(255,255,255,0.01)", border: "1px solid rgba(255,255,255,0.04)", padding: "10px 14px", borderRadius: "12px" }}>
                        <span style={{ fontSize: "0.8rem", fontWeight: "700" }}>Hallucination Filter</span>
                        <span style={{ fontSize: "0.75rem", background: "rgba(16, 185, 129, 0.15)", color: "#10b981", padding: "2px 8px", borderRadius: "4px", fontWeight: "700" }}>ENABLED</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {uploadError && <div className={styles.errorBox}>{uploadError}</div>}
            </div>

            {/* Panel 3: Chat Container */}
            <div className={styles.chatPanel}>
              <div className={styles.chatHistory}>
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`${styles.messageWrapper} ${msg.sender === "user" ? styles.userWrapper : styles.aiWrapper}`}
                  >
                    {msg.sender === "ai" && (
                      <div className={styles.avatar}>🤖</div>
                    )}

                    <div className={styles.messageContent}>
                      {/* Multi-stage Thinking Logs */}
                      {msg.thinkingStages && msg.thinkingStages.length > 0 && (
                        <div className={styles.thinkingAccordion}>
                          <div className={styles.thinkingHeader}>
                            <div className={styles.miniSpinner}></div>
                            <span>Nemotron Reasoning Stage</span>
                          </div>
                          <div className={styles.thinkingStagesList}>
                            {msg.thinkingStages.map((stage, idx) => (
                              <div key={idx} className={styles.stageItem}>
                                {stage}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {msg.isThinking && !msg.text && (
                        <div className={styles.thinkingSkeleton}>
                          <div className="shimmer" style={{ width: "80%", height: "14px", borderRadius: "4px", marginBottom: "8px" }}></div>
                          <div className="shimmer" style={{ width: "60%", height: "14px", borderRadius: "4px" }}></div>
                        </div>
                      )}

                      {msg.text && (
                        <div className={styles.bubbleText}>
                          {msg.text.includes("[WARNING: Response failed hallucination filter]") || msg.text.includes("⚠️ [WARNING:") ? (
                            <span style={{ color: "#f87171" }}>{msg.text}</span>
                          ) : (
                            msg.text
                          )}
                        </div>
                      )}
                    </div>

                    {msg.sender === "user" && (
                      <div className={styles.avatarUser}>👤</div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Form input */}
              <form className={styles.inputArea} onSubmit={sendMessage}>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={sessionId ? "Ask anything about the document..." : "Upload document to unlock chat..."}
                  disabled={!sessionId || isChatting}
                  className={styles.chatInput}
                />
                <button
                  type="submit"
                  disabled={!sessionId || !input.trim() || isChatting}
                  className={styles.sendButton}
                >
                  Send
                </button>
              </form>
            </div>
          </>
        )}

        {activeTab === "space" && (
          /* Panel 2: Interactive 3D Embedding Space */
          <div className={styles.visualizerPanel} style={{ height: "100%" }}>
            <div className={styles.cardHeader}>
              <h3>3D Embedding Space Projection</h3>
              <p>Dimensional view of document semantic chunks. Hover nodes to pause rotation and inspect values.</p>
            </div>
            <div className={styles.canvasContainer}>
              {nodes.length > 0 ? (
                <EmbeddingSpace nodes={nodes} retrievedChunks={retrievedChunks} />
              ) : (
                <div className={styles.emptyVisualizer}>
                  <div className={styles.visualizerPlaceholderIcon}>🌐</div>
                  <p className={styles.visualizerPlaceholderText}>Upload a document in the Conversation tab to project chunks in 3D space</p>
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
