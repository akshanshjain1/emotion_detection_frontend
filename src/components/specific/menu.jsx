import { Download, FileText, Code } from "lucide-react";
import { useState } from "react";
import CodeViewer from "./codeviewer";

export default function Menu({viewcode=false,setviewcode}) {
    
    const handleViewcode=()=>{
        setviewcode(prev=>!prev)
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
        <button className="flex items-center gap-3 bg-blue-600 hover:bg-blue-700 text-white text-md font-semibold py-3 px-5 rounded-lg transition"
        onClick={handleDownload}>
          <Download className="w-5 h-5" />
          <span>download model</span>
        </button>

        <button className="flex items-center gap-3 bg-green-600 hover:bg-green-700 text-white text-md font-semibold py-3 px-5 rounded-lg transition">
          <Download className="w-5 h-5" />
          <a href="https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess">
          download dataset    </a> 
        </button>

        <button className="flex items-center gap-3 bg-purple-600 hover:bg-purple-700 text-white text-md font-semibold py-3 px-5 rounded-lg transition"
        onClick={setviewcode}>
          <Code className="w-5 h-5" />
          <span>view code</span>
        </button>
      </div>
      
    </div>
  );
}
