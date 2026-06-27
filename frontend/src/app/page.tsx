"use client";

import React, { useState, useRef, useEffect } from "react";
import styles from "./page.module.css";

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
      
      // Update chat message indicating success
      setMessages((prev) => [
        ...prev,
        {
          id: `upload-${Date.now()}`,
          sender: "ai",
          text: `📄 **${selectedFile.name}** was successfully indexed into knowledge base! What would you like to know about it?`,
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
            const dataContent = line.slice(6).trim();
            if (!dataContent) continue;

            // Check if the data is a thinking/reasoning stage
            if (dataContent.startsWith("🔍") || dataContent.startsWith("🧠") || dataContent.startsWith("📝")) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === messageId
                    ? {
                        ...msg,
                        thinkingStages: [...(msg.thinkingStages || []), dataContent],
                      }
                    : msg
                )
              );
            } else {
              // Append actual content text
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === messageId
                    ? {
                        ...msg,
                        isThinking: false,
                        text: msg.text + dataContent,
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
          <span className={styles.logoBadge}>AI</span>
          <h1>Cognitive RAG Hub</h1>
        </div>
        <p className={styles.headerSubtitle}>Session-bound Dynamic Context Retrieval & Reasoning Engine</p>
      </header>

      <section className={styles.workspace}>
        {/* Left Side: Upload Panel */}
        <div className={styles.leftPanel}>
          <div className={styles.cardHeader}>
            <h3>Knowledge Injector</h3>
            <p>Upload a source document to start matching queries</p>
          </div>

          <div
            className={`${styles.dragArea} ${dragActive ? styles.dragActive : ""} ${file ? styles.fileLoaded : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              ref={fileInputRef}
              className={styles.hiddenInput}
              onChange={handleFileChange}
              accept=".pdf,.txt"
            />

            {!file ? (
              <div className={styles.uploadPrompt}>
                <div className={styles.pulseIcon}>📥</div>
                <p className={styles.primaryText}>Drag & drop document here</p>
                <p className={styles.secondaryText}>Only PDF or TXT up to 10MB</p>
                <button type="button" className={styles.browseButton} onClick={triggerBrowse}>
                  Browse Files
                </button>
              </div>
            ) : (
              <div className={styles.fileDetails}>
                <div className={styles.fileIcon}>📄</div>
                <p className={styles.fileName}>{file.name}</p>
                <p className={styles.fileSize}>{(file.size / 1024).toFixed(1)} KB</p>
                {isUploading ? (
                  <div className={styles.processingIndicator}>
                    <div className={styles.spinner}></div>
                    <span>Parsing and embedding content...</span>
                  </div>
                ) : (
                  <div className={styles.statusSuccess}>
                    <span className={styles.successCheck}>✓</span> Indexed and Ready
                  </div>
                )}
              </div>
            )}
          </div>

          {uploadError && <div className={styles.errorBox}>{uploadError}</div>}
          
          <div className={styles.tipsBox}>
            <h4>💡 How to use</h4>
            <ul>
              <li>Upload files to dynamically construct a vectorized space.</li>
              <li>Queries leverage cosine similarity lookup before consulting the LLM.</li>
              <li>Every conversation is session-bound. Start fresh by uploading a new file.</li>
            </ul>
          </div>
        </div>

        {/* Right Side: Chat Container */}
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
                        <span>Nemotron Reasoning Pipeline</span>
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
                      {msg.text}
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
      </section>
    </main>
  );
}
