import { Download, FileText, Code } from "lucide-react";
import { useState } from "react";
import CodeViewer from "./codeviewer";

export default function Menu({ viewcode = false, setviewcode }) {

  const handleViewcode = () => {
    setviewcode(prev => !prev)
  }
  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = "/speech_emotion_recognition_model_optimized (1).h5"; // Adjust path if needed
    link.download = "speech_emotion_recognition_model.h5"; // Set download filename
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="w-64 h-full bg-gray-900 text-white p-6 shadow-lg flex flex-col gap-6 ">
      {/* Header */}
      <h2 className="text-xl font-bold text-gray-200 mt-[13%] pt-[3%]">Menu</h2>

      {/* Options */}
      <div className="flex flex-col gap-4">
        <button
          className="flex items-center gap-3 bg-white/10 hover:bg-white/20 text-white text-md font-semibold py-3 px-5 rounded-xl transition-all duration-300 border border-white/10 backdrop-blur-md shadow-lg hover:shadow-blue-500/20 group"
          onClick={handleDownload}
        >
          <Download className="w-5 h-5 text-blue-400 group-hover:text-blue-300 transition-colors" />
          <span>Download Model</span>
        </button>

        <a
          href="https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 bg-white/10 hover:bg-white/20 text-white text-md font-semibold py-3 px-5 rounded-xl transition-all duration-300 border border-white/10 backdrop-blur-md shadow-lg hover:shadow-green-500/20 group"
        >
          <Download className="w-5 h-5 text-green-400 group-hover:text-green-300 transition-colors" />
          <span>Download Dataset</span>
        </a>

        <button
          className="flex items-center gap-3 bg-white/10 hover:bg-white/20 text-white text-md font-semibold py-3 px-5 rounded-xl transition-all duration-300 border border-white/10 backdrop-blur-md shadow-lg hover:shadow-purple-500/20 group"
          onClick={handleViewcode}
        >
          <Code className="w-5 h-5 text-purple-400 group-hover:text-purple-300 transition-colors" />
          <span>View Code</span>
        </button>
      </div>

    </div>
  );
}
