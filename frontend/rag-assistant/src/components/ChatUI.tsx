
import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Role = "assistant" | "user" | "system";

interface Message {
  role: Role;
  content: string;
  sources?: string[]; // optional list of source filenames
}

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const ChatUI: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "👋 Hi! Upload a doc or ask a question." },
  ]);
  const [input, setInput] = useState<string>("");
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to bottom whenever messages update
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const sendQuery = async (query: string) => {
    setIsTyping(true);
    try {
      const formData = new FormData();
      formData.append("query", query);

      const res = await fetch(`${API_BASE}/query/`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer ?? "No answer returned.",
        sources: data.sources ?? [],
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ Error contacting backend: ${err?.message || err}`,
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const newMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, newMessage]);
    const q = input;
    setInput("");
    await sendQuery(q);
  };

  const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

    const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    try {
        const formData = new FormData();
        formData.append("file", file); // <-- single-file key

        const res = await fetch(`${API_BASE}/upload/`, {
        method: "POST",
        body: formData, // don't set Content-Type yourself
        });

        if (!res.ok) throw new Error(`Upload failed with ${res.status}`);
        const data = await res.json();

        setMessages((prev) => [...prev, { role: "assistant", content: `✅ ${data.message}` }]);
        setFile(null);
    } catch (err: any) {
        setMessages((prev) => [...prev, { role: "assistant", content: `⚠️ Upload error: ${err?.message || err}` }]);
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

      {/* Chat Area */}
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
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>

              {/* Sources (if any) */}
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

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="px-4 py-3 bg-gray-700 rounded-2xl rounded-bl-none text-gray-300 text-sm animate-pulse">
              Assistant is typing...
            </div>
          </div>
        )}
      </div>

      {/* Input / Upload Bar */}
      <div className="p-4 bg-gray-800/70 backdrop-blur-md flex items-center gap-3 border-t border-gray-700">
        {/* File chooser */}
        <label className="flex items-center gap-2 cursor-pointer text-sm bg-gray-700 px-3 py-2 rounded-lg">
        <input
            type="file"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            accept=".txt,.md,.pdf,.html"
        />
        {file ? `Selected: ${file.name}` : "Choose file"}
        </label>

        {/* upload button */}
        <button
        onClick={handleUpload}
        disabled={!file || isUploading}
        className={`${!file || isUploading ? "bg-gray-600 cursor-not-allowed" : "bg-blue-500 hover:bg-blue-600"} text-white px-3 py-2 rounded-lg shadow-md transition text-sm`}
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

        <button
          onClick={handleSend}
          disabled={isTyping}
          className={`${
            isTyping ? "bg-gray-600 cursor-not-allowed" : "bg-green-500 hover:bg-green-600"
          } text-white px-4 py-2 rounded-lg shadow-md transition`}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatUI;
