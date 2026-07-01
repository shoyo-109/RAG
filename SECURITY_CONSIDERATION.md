# Security Considerations: Advanced RAG Pipeline

This document details the security layers implemented inside the Advanced RAG pipeline to prevent prompt injections, protect personal information, and block harmful generations.

---

## 🛡️ Input Sanitization & Attack Mitigation

### 1. Prompt Injection Block
- **Regex Patterns**: Scans user inputs against common injection signatures (e.g. `ignore all previous instructions`, `bypass all restrictions`, `pretend you are`).
- **Sanitization Actions**: Blocks execution immediately or escapes specific syntax delimiters (e.g. replacing `{` with `{ {`) to prevent LangChain templates from being hijacked.

---

## 🔒 PII Masking & Data Leakage Protection

### 1. PII Regex Engine
- The pipeline scans query inputs and outputs for sensitive personal identifiers:
  - **Emails**: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
  - **Phones**: `\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`
  - **SSNs**: `\b\d{3}-\d{2}-\d{4}\b`
  - **Credit Cards**: `\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b`
  - **IP Addresses**: `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`
- **Masking Mechanism**: Replaces matches with redact tags (e.g. `[EMAIL REDACTED]`) before writing to the semantic cache or passing text to internal prompts.

---

## 🕵️ Output Verification & Content Safety

### 1. Hallucination Reflector
- A separate LLM validation step compares the output of the model against the raw retrieved chunks.
- If the output contains assumptions or statements not explicitly grounded in the database context, a warning flag is prepended to alert the user.

### 2. Output Validator
- Checks outgoing responses for leaked secrets, API keys, passwords, or instructions explaining malicious actions (such as hacking or security bypasses). Any matched response is automatically replaced with `[CONTENT BLOCKED]`.

---

## 🌐 Network & Infrastructure Best Practices

1. **API Key Management**: Never hardcode keys. Store secrets in environment variables (`.env`, Render environment variables) for `NVIDIA_API_KEY`, `OPENAI_API_KEY`, and `LANGCHAIN_API_KEY`.
2. **CORS Restrictions**: Currently configured to allow `"*"` in development. For production deployments, restrict FastAPI CORS middleware settings (`allow_origins`) to whitelist only your production Vercel frontend URL.
3. **SSL/TLS Encryption**: Ensure all API requests route through `https` protocols in production to avoid man-in-the-middle sniffing of document contents.
