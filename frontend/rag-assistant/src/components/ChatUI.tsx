import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Define roles for messages in the chat
type Role = "assistant" | "user" | "system";

// Shape of a message in the chat
interface Message {
  role: Role;
  content: string;
  sources?: string[]; // Optional list of source filenames
}

// Base URL for backend API (from env or defaults to localhost)
const API_BASE =
  (import.meta as any).env?.VITE_API_BASE ||
  (import.meta as any).env?.VITE_API_URL ||
  "http://localhost:8000";

const ChatUI: React.FC = () => {
  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "👋 Hi! Upload a doc or ask a question." },
  ]);
  const [input, setInput] = useState<string>(""); // User’s input text
  const [isTyping, setIsTyping] = useState<boolean>(false); // Assistant typing indicator
  const [file, setFile] = useState<File | null>(null); // Uploaded file
  const [isUploading, setIsUploading] = useState(false); // Uploading state
  const chatContainerRef = useRef<HTMLDivElement>(null); // Ref to scroll chat

  // Auto-scroll to bottom when messages or typing state changes
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  // Fetch wrapper with timeout (to avoid hanging requests)
  const fetchWithTimeout = async (
    input: RequestInfo | URL,
    init: RequestInit & { timeoutMs?: number } = {}
  ) => {
    const { timeoutMs = 60000, ...rest } = init; // default timeout 60s
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(input, { ...rest, signal: controller.signal });
      return res;
    } finally {
      clearTimeout(id);
    }
  };

  // Send user query to backend and get assistant response
  const sendQuery = async (query: string) => {
    setIsTyping(true);
    try {
      const formData = new FormData();
      formData.append("query", query);

      const res = await fetchWithTimeout(`${API_BASE}/query/`, {
        method: "POST",
        body: formData,
        timeoutMs: 60000, // safety timeout
      });

      if (!res.ok) throw new Error(`Backend returned ${res.status}`);

      const data = await res.json();
      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer ?? "No answer returned.",
        sources: data.sources ?? [],
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      // Handle errors (including timeout)
      const msg =
        err?.name === "AbortError"
          ? "⚠️ Request timed out. Try again or reduce context size."
          : `⚠️ Error contacting backend: ${err?.message || err}`;
      setMessages((prev) => [...prev, { role: "assistant", content: msg }]);
    } finally {
      setIsTyping(false);
    }
  };

  // Handle sending a message from the input box
  const handleSend = async () => {
    if (!input.trim()) return;
    const newMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, newMessage]);
    const q = input;
    setInput("");
    await sendQuery(q);
  };

  // Allow Enter key to send messages
  const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Handle uploading a file to backend
  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      // 120s timeout since embedding models may need to download
      const res = await fetchWithTimeout(`${API_BASE}/upload/`, {
        method: "POST",
        body: formData,
        timeoutMs: 120000,
      });

      if (!res.ok) throw new Error(`Upload failed with ${res.status}`);
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `✅ ${data.message}` },
      ]);
      setFile(null);
    } catch (err: any) {
      // Handle upload errors
      const msg =
        err?.name === "AbortError"
          ? "⚠️ Upload is taking too long (first-time model warmup). It will finish soon; try again in a bit."
          : `⚠️ Upload error: ${err?.message || err}`;
      setMessages((prev) => [...prev, { role: "assistant", content: msg }]);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="p-4 bg-gray-800/70 backdrop-blur-md shadow-md text-lg font-semibold text-center border-b border-gray-700">
        RAG Knowledge Assistant
      </header>

      {/* Chat messages area */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800"
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${
              msg.role === "assistant" ? "justify-start" : "justify-end"
            }`}
          >
            <div
              className={`px-4 py-3 rounded-2xl max-w-2xl shadow-md text-sm leading-relaxed ${
                msg.role === "assistant"
                  ? "bg-gray-700 text-white rounded-bl-none"
                  : "bg-green-500 text-white rounded-br-none"
              }`}
            >
              {/* Render Markdown for better formatting */}
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>

              {/* Show document sources if available */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 text-xs text-gray-300">
                  <div className="opacity-80">Sources:</div>
                  <ul className="list-disc list-inside">
                    {msg.sources.map((s, i) => (
                      <li key={`${s}-${i}`}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Assistant typing indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="px-4 py-3 bg-gray-700 rounded-2xl rounded-bl-none text-gray-300 text-sm animate-pulse">
              Assistant is typing...
            </div>
          </div>
        )}
      </div>

      {/* Input and Upload controls */}
      <div className="p-4 bg-gray-800/70 backdrop-blur-md flex items-center gap-3 border-t border-gray-700">
        {/* File selector */}
        <label className="flex items-center gap-2 cursor-pointer text-sm bg-gray-700 px-3 py-2 rounded-lg">
          <input
            type="file"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            accept=".txt,.md,.pdf,.html"
          />
          {file ? `Selected: ${file.name}` : "Choose file"}
        </label>

        {/* Upload button */}
        <button
          onClick={handleUpload}
          disabled={!file || isUploading}
          className={`${
            !file || isUploading
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-500 hover:bg-blue-600"
          } text-white px-3 py-2 rounded-lg shadow-md transition text-sm`}
        >
          {isUploading ? "Uploading..." : "Upload"}
        </button>

        {/* Query input */}
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask me anything about your docs..."
          className="flex-1 p-3 bg-gray-700 rounded-lg outline-none text-white placeholder-gray-400"
        />

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={isTyping}
          className={`${
            isTyping
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-green-500 hover:bg-green-600"
          } text-white px-4 py-2 rounded-lg shadow-md transition`}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatUI;
