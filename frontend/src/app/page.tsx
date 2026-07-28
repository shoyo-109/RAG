"use client";

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import styles from "./page.module.css";
import EmbeddingSpace, { ChunkNode } from "./EmbeddingSpace";

interface Message {
  id: string;
  sender: "user" | "ai";
  text: string;
  thinkingStages?: string[];
  isThinking?: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; size: number }[]>([]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStep, setUploadStep] = useState<number>(1);
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

  // Trigger warmup request when page loads
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/warmup`).catch((err) => {
      console.warn("Backend warmup trigger failed:", err);
    });
  }, []);

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

  // Reset active session
  const resetSession = () => {
    setUploadedFiles([]);
    setSessionId(null);
    setNodes([]);
    setRetrievedChunks([]);
    setUploadError(null);
    setInput("");
    setMessages([
      {
        id: "initial",
        sender: "ai",
        text: "👋 Welcome! Start by dropping a PDF or TXT file on the panel to the left. Once indexed, we can start discussing it.",
      },
    ]);
  };

  // Upload and Index Document via FastAPI
  const processAndUploadFile = async (selectedFile: File) => {
    // 50MB file size limit check in frontend
    if (selectedFile.size > 50 * 1024 * 1024) {
      setUploadError("File size exceeds the maximum limit of 50MB.");
      return;
    }

    const ext = selectedFile.name.split(".").pop()?.toLowerCase();
    if (ext !== "pdf" && ext !== "txt") {
      setUploadError("Only PDF and TXT documents are supported.");
      return;
    }

    setUploadError(null);
    setIsUploading(true);
    setUploadStep(1);

    // Simulated progress stepper updates
    const stepInterval = setInterval(() => {
      setUploadStep((prev) => (prev < 4 ? prev + 1 : prev));
    }, 1500);

    const isAppend = sessionId !== null;
    if (!isAppend) {
      setNodes([]);
      setRetrievedChunks([]);
      setUploadedFiles([{ name: selectedFile.name, size: selectedFile.size }]);
    } else {
      setUploadedFiles((prev) => [...prev, { name: selectedFile.name, size: selectedFile.size }]);
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (isAppend && sessionId) {
      formData.append("session_id", sessionId);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
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
          text: isAppend
            ? `➕ Added **${selectedFile.name}** to the active knowledge base. Custom HNSW and BM25 indices updated! Try asking questions.`
            : `📄 **${selectedFile.name}** was successfully indexed into advanced knowledge base! Custom HNSW index ready. Try asking questions.`,
        },
      ]);
    } catch (err: any) {
      setUploadError(err.message || "An error occurred during file upload.");
      if (isAppend) {
        setUploadedFiles((prev) => prev.slice(0, -1));
      } else {
        setUploadedFiles([]);
      }
    } finally {
      clearInterval(stepInterval);
      setUploadStep(4);
      setIsUploading(false);
    }
  };

  // Send Chat message and handle SSE response stream
  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isChatting || isUploading) return;

    if (!sessionId) {
      setMessages((prev) => [
        ...prev,
        {
          id: `msg-${Date.now()}`,
          sender: "ai",
          text: "⚠️ Please upload a PDF or TXT document first on the left panel before asking questions!",
        },
      ]);
      return;
    }


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
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
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
            // Check if final presentation-transformed markdown output is received
            else if (trimmedContent.startsWith("final_transformed:")) {
              try {
                const transformedText = JSON.parse(trimmedContent.slice(18));
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === messageId
                      ? {
                        ...msg,
                        isThinking: false,
                        text: transformedText,
                      }
                      : msg
                  )
                );
              } catch (e) {
                console.error("Failed to parse final_transformed payload", e);
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

              {uploadedFiles.length === 0 ? (
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
                    <p className={styles.secondaryText}>PDF or TXT up to 50MB</p>
                    <button type="button" className={styles.browseButton}>
                      Browse Files
                    </button>
                  </div>
                </div>
              ) : (
                <div className={styles.fileLoadedContainer}>
                  {/* File Metadata Details */}
                  <div className={styles.fileListContainer}>
                    {uploadedFiles.map((f, idx) => (
                      <div key={idx} className={styles.fileListItem}>
                        <div className={styles.fileItemIcon}>📄</div>
                        <div className={styles.fileItemInfo}>
                          <p className={styles.fileName}>{f.name}</p>
                          <p className={styles.fileSize}>{(f.size / 1024).toFixed(1)} KB</p>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Actions for active session */}
                  <div className={styles.fileActions}>
                    <button
                      type="button"
                      onClick={triggerBrowse}
                      className={styles.appendButton}
                      disabled={isUploading}
                    >
                      {isUploading ? "Uploading..." : "➕ Add Doc"}
                    </button>
                    <button
                      type="button"
                      onClick={resetSession}
                      className={styles.resetButton}
                      disabled={isUploading}
                    >
                      Reset Session
                    </button>
                    <input
                      type="file"
                      ref={fileInputRef}
                      className={styles.hiddenInput}
                      onChange={handleFileChange}
                      accept=".pdf,.txt"
                    />
                  </div>

                  {/* Dynamic Pipeline Progress Stepper */}
                  <div className={styles.stepperContainer}>
                    <div className={`${styles.stepItem} ${!isUploading ? styles.stepDone : uploadStep === 1 ? styles.stepActive : uploadStep > 1 ? styles.stepDone : styles.stepPending}`}>
                      <span className={styles.stepIcon}>
                        {uploadStep === 1 && isUploading ? (
                          <div className={`${styles.miniSpinner} ${styles.stepSpinner}`}></div>
                        ) : (!isUploading || uploadStep > 1) ? "✓" : "⚡"}
                      </span>
                      <span>1. Extracting Document Text</span>
                      <span className={styles.stepBadge}>
                        {!isUploading || uploadStep > 1 ? "Done" : uploadStep === 1 ? "Active" : "Pending"}
                      </span>
                    </div>
                    
                    <div className={`${styles.stepItem} ${!isUploading ? styles.stepDone : uploadStep === 2 ? styles.stepActive : uploadStep > 2 ? styles.stepDone : styles.stepPending}`}>
                      <span className={styles.stepIcon}>
                        {uploadStep === 2 && isUploading ? (
                          <div className={`${styles.miniSpinner} ${styles.stepSpinner}`}></div>
                        ) : (!isUploading || uploadStep > 2) ? "✓" : "⚙️"}
                      </span>
                      <span>2. Semantic Percentile Chunking</span>
                      <span className={styles.stepBadge}>
                        {!isUploading || uploadStep > 2 ? "Done" : uploadStep === 2 ? "Active" : "Pending"}
                      </span>
                    </div>

                    <div className={`${styles.stepItem} ${!isUploading ? styles.stepDone : uploadStep === 3 ? styles.stepActive : uploadStep > 3 ? styles.stepDone : styles.stepPending}`}>
                      <span className={styles.stepIcon}>
                        {uploadStep === 3 && isUploading ? (
                          <div className={`${styles.miniSpinner} ${styles.stepSpinner}`}></div>
                        ) : (!isUploading || uploadStep > 3) ? "✓" : "🕸️"}
                      </span>
                      <span>3. Indexing into Qdrant Cloud</span>
                      <span className={styles.stepBadge}>
                        {!isUploading || uploadStep > 3 ? "Done" : uploadStep === 3 ? "Active" : "Pending"}
                      </span>
                    </div>

                    <div className={`${styles.stepItem} ${!isUploading ? styles.stepDone : uploadStep === 4 ? styles.stepActive : styles.stepPending}`}>
                      <span className={styles.stepIcon}>
                        {uploadStep === 4 && isUploading ? (
                          <div className={`${styles.miniSpinner} ${styles.stepSpinner}`}></div>
                        ) : !isUploading ? "✓" : "🗂️"}
                      </span>
                      <span>4. Rebuilding BM25 Lexical Index</span>
                      <span className={styles.stepBadge}>
                        {!isUploading ? "Done" : uploadStep === 4 ? "Active" : "Pending"}
                      </span>
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
                        <div className={styles.statValue}>Qdrant</div>
                        <div className={styles.statLabel}>Cloud Store</div>
                      </div>
                    </div>
                  )}

                  {/* Interactive RAG Parameter Sliders */}
                  {!isUploading && sessionId && (
                    <div className={styles.parameterControls}>
                      <div className={styles.parameterSlider}>
                        <div className={styles.sliderLabel}>
                          <span>Context Chunks (K)</span>
                          <span className={styles.sliderValue}>{topK}</span>
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

                      <div className={styles.hallucinationFilterBadge}>
                        <span className={styles.filterTitle}>Hallucination Filter</span>
                        <span className={styles.filterStatus}>ENABLED</span>
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
                          <div className={`shimmer ${styles.skeletonLineLong}`}></div>
                          <div className={`shimmer ${styles.skeletonLineShort}`}></div>
                        </div>
                      )}

                      {msg.text && (
                        <div className={styles.bubbleText}>
                          {msg.text.includes("[WARNING: Response failed hallucination filter]") || msg.text.includes("⚠️ [WARNING:") ? (
                            <span className={styles.warningText}>
                              <ReactMarkdown>{msg.text}</ReactMarkdown>
                            </span>
                          ) : (
                            <ReactMarkdown>{msg.text}</ReactMarkdown>
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
                  placeholder={sessionId ? "Ask anything about the document..." : "Upload a document to enable answering..."}
                  disabled={isChatting || isUploading}
                  className={styles.chatInput}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isChatting || isUploading}
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
          <div className={`${styles.visualizerPanel} ${styles.visualizerPanelFullHeight}`}>
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
