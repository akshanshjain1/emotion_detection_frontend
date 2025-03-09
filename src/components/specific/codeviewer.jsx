import { useState } from "react";
import { Copy, Check, X } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export default function CodeViewer({ code, language = "javascript", onClose }) {
  const [copied, setCopied] = useState(false);

  // Copy to clipboard
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 flex justify-center items-center bg-black/50">
      <div className="relative w-[90%] max-w-3xl bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
        {/* Header */}
        <div className="flex justify-between items-center px-4 py-2 bg-gray-800">
          <span className="text-sm font-semibold text-gray-300">{language.toUpperCase()}</span>
          <div className="flex gap-3">
            {/* Close Button */}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition"
            >
              <X className="w-6 h-6" />
            </button>
            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-gray-300 hover:text-white transition"
            >
              {copied ? <Check className="w-5 h-5 text-green-400" /> : <Copy className="w-5 h-5" />}
              <span className="text-sm">{copied ? "Copied!" : "Copy"}</span>
            </button>
          </div>
        </div>

        {/* Scrollable Code Block */}
        <div className="max-h-[70vh] overflow-auto">
          <SyntaxHighlighter language={language} style={oneDark} className="p-4 text-sm">
            {code}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  );
}
