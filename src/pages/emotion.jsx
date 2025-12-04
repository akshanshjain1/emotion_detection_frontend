import { useState, useRef, useEffect } from "react";
import { UploadCloud, Mic, Play, Trash, Save, Sparkles, StopCircle, Brain, Heart, Lightbulb, RefreshCw, CheckCircle2 } from "lucide-react";
import "./emotion.css";
import axios from "axios";
import EmojiSwitcher from "../components/ui/emoji";
import AudioVisualizer from "../components/ui/AudioVisualizer";
import { generateGroqStream, getWellnessTips, getAffirmation } from "../utils/api";

export default function EmotionDetection({ menu = false, setmenu }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recordTime, setRecordTime] = useState(0);
  const [audioURL, setAudioURL] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [submitDisabled, setSubmitDisabled] = useState(true);
  const [DetectedEmotion, setDetectedEmotion] = useState("");
  const [llmResponse, setLlmResponse] = useState("");
  const [wellnessTips, setWellnessTips] = useState("");
  const [affirmation, setAffirmation] = useState("");
  const [llmLoading, setLlmLoading] = useState(false);
  const [tipsLoading, setTipsLoading] = useState(false);
  const [affirmationLoading, setAffirmationLoading] = useState(false);
  const [loading, setisloading] = useState(false);
  const [stream, setStream] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);

  const groqApiKey = import.meta.env.VITE_GROQ_API_KEY || "";

  const mediaRecorderRef = useRef(null);
  const timerRef = useRef(null);
  const audioRef = useRef(null);
  const block1Ref = useRef(null);
  const block2Ref = useRef(null);
  const block3Ref = useRef(null);

  useEffect(() => {
    const scrollToBlock = (ref) => {
      if (ref.current) {
        ref.current.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    };
    // Only auto-scroll for steps 2 and 3 (not step 1 - we want to see header first)
    if (currentStep === 2) scrollToBlock(block2Ref);
    else if (currentStep === 3) scrollToBlock(block3Ref);
  }, [currentStep, llmResponse]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file && (file.type === "audio/wav" || file.type === "audio/mpeg")) {
      setSelectedFile(file);
      setAudioBlob(null);
      setAudioURL(null);
      setUploading(true);
      setSubmitDisabled(true);
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        setUploadProgress(progress);
        if (progress >= 100) {
          clearInterval(interval);
          setUploading(false);
          setSubmitDisabled(false);
        }
      }, 300);
    } else {
      alert("Please upload a valid WAV or MP3 file!");
    }
  };

  const handleRecord = async () => {
    if (!recording) {
      try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setStream(audioStream);
        const mediaRecorder = new MediaRecorder(audioStream, { mimeType: "audio/webm" });
        mediaRecorderRef.current = mediaRecorder;
        let audioChunks = [];
        mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/mpeg" });
          const audioUrl = URL.createObjectURL(audioBlob);
          setAudioBlob(audioBlob);
          setAudioURL(audioUrl);
          setRecordTime(0);
          clearInterval(timerRef.current);
          setSubmitDisabled(false);
          audioStream.getTracks().forEach(track => track.stop());
          setStream(null);
        };
        mediaRecorder.start();
        setRecording(true);
        timerRef.current = setInterval(() => setRecordTime((prev) => prev + 1), 1000);
      } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Could not access microphone.");
      }
    } else {
      mediaRecorderRef.current?.stop();
      setRecording(false);
    }
  };

  const convertWebMtoWAV = async (webmBlob) => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsArrayBuffer(webmBlob);
      fileReader.onloadend = async () => {
        try {
          const audioContext = new AudioContext();
          const audioBuffer = await audioContext.decodeAudioData(fileReader.result);
          const wavBlob = encodeWAV(audioBuffer);
          resolve(wavBlob);
        } catch (error) {
          reject(error);
        }
      };
    });
  };

  const encodeWAV = (audioBuffer) => {
    const numOfChan = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const samples = audioBuffer.getChannelData(0);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numOfChan, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numOfChan * 2, true);
    view.setUint16(32, numOfChan * 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);
    const floatTo16BitPCM = (output, offset, input) => {
      for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
    };
    floatTo16BitPCM(view, 44, samples);
    return new Blob([buffer], { type: "audio/wav" });
  };

  const handleSaveRecording = async () => {
    if (audioBlob) {
      try {
        const wavBlob = await convertWebMtoWAV(audioBlob);
        const newFile = new File([wavBlob], "recorded_audio.wav", { type: "audio/wav" });
        setSelectedFile(newFile);
        alert("Recording saved!");
      } catch (error) {
        alert("Conversion failed!");
      }
    }
  };

  const handleSubmit = async () => {
    if (!groqApiKey) {
      alert("Please set VITE_GROQ_API_KEY in your .env file!");
      return;
    }
    setisloading(true);
    setCurrentStep(2);
    setLlmResponse("");
    setWellnessTips("");
    setAffirmation("");

    const formData = new FormData();
    if (selectedFile) {
      formData.append("file", selectedFile);
    } else if (audioBlob) {
      formData.append("file", audioBlob, "recorded_audio.wav");
    }

    try {
      const response = await axios.post(
        "https://emotion-detection-87as.onrender.com/predict/",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      const emotionRaw = response.data.predicted_emotion.slice(4).toUpperCase();
      setDetectedEmotion(emotionRaw);

      setTimeout(() => setCurrentStep(3), 1500);

      setLlmLoading(true);
      await generateGroqStream(emotionRaw, groqApiKey, (chunk) => {
        setLlmResponse((prev) => prev + chunk);
      }, 'comprehensive');
      setLlmLoading(false);

      setTipsLoading(true);
      const tips = await getWellnessTips(emotionRaw, groqApiKey);
      setWellnessTips(tips);
      setTipsLoading(false);

      setAffirmationLoading(true);
      const affirm = await getAffirmation(emotionRaw, groqApiKey);
      setAffirmation(affirm);
      setAffirmationLoading(false);
    } catch (error) {
      alert(`Error: ${error.message}`);
      setisloading(false);
    }
  };

  const handleRetry = () => {
    setAudioBlob(null);
    setAudioURL(null);
    setSelectedFile(null);
    setSubmitDisabled(true);
    setDetectedEmotion("");
    setLlmResponse("");
    setWellnessTips("");
    setAffirmation("");
    setisloading(false);
    setLlmLoading(false);
    setTipsLoading(false);
    setAffirmationLoading(false);
    setCurrentStep(1);
  };

  const StepBadge = ({ number, active, completed }) => (
    <div className={`step-badge ${!active && !completed ? 'step-badge-inactive' : ''} ${completed ? 'bg-green-500' : ''}`}>
      {completed ? <CheckCircle2 className="w-6 h-6" /> : number}
    </div>
  );

  const TimelineConnector = ({ active }) => (
    <div className="flex justify-center py-4">
      <div className={`w-1 h-16 rounded-full transition-all duration-500 ${active ? 'bg-gradient-to-b from-indigo-500 to-pink-500' : 'bg-gray-300'}`} />
    </div>
  );

  return (
    <div className="min-h-screen w-full overflow-y-auto bg-gradient-to-br from-pink-100 via-purple-100 to-indigo-200 relative py-8">
      {/* Background Blobs */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-5%] left-[-5%] w-[30%] h-[30%] bg-gradient-to-br from-pink-300 to-purple-400 opacity-30 blur-3xl animate-blob"></div>
        <div className="absolute top-[20%] right-[-5%] w-[35%] h-[35%] bg-gradient-to-br from-indigo-300 to-blue-400 opacity-30 blur-3xl animate-blob" style={{ animationDelay: '2s' }}></div>
        <div className="absolute bottom-[-5%] left-[20%] w-[30%] h-[30%] bg-gradient-to-br from-purple-300 to-pink-400 opacity-30 blur-3xl animate-blob" style={{ animationDelay: '4s' }}></div>
      </div>

      <div className="z-10 relative w-full max-w-4xl mx-auto px-4 flex flex-col items-center">

        {/* ===== HEADER / TOPIC TITLE ===== */}
        <div className="mb-12 text-center">
          <p className="text-sm uppercase tracking-[0.3em] text-indigo-600 font-semibold mb-4">AI-Powered Analysis</p>
          <h1 className="text-4xl md:text-6xl font-black mb-4 animate-text-glow tracking-tight leading-tight">
            Emotion Detection
            <br />
            <span className="text-3xl md:text-5xl">Using Speech</span>
          </h1>
          <p className="text-lg text-gray-600 font-medium max-w-lg mx-auto">
            Discover what your voice reveals about your emotions and receive personalized AI insights
          </p>
        </div>

        {/* ===== BLOCK 1: RECORD/UPLOAD ===== */}
        <div ref={block1Ref} className={`w-full block-container ${currentStep === 1 ? 'active' : ''}`}>
          <div className="flex items-center gap-4 mb-6">
            <StepBadge number={1} active={currentStep === 1} completed={currentStep > 1} />
            <h2 className="text-2xl font-bold text-gray-800">Record or Upload Audio</h2>
          </div>

          <div className="glass-premium rounded-3xl p-8 shadow-xl">
            <div className="w-full h-40 flex items-center justify-center relative mb-6">
              {recording ? (
                <AudioVisualizer stream={stream} isRecording={recording} />
              ) : audioURL ? (
                <div className="w-full h-32 bg-gradient-to-r from-indigo-100 to-pink-100 rounded-2xl flex items-center justify-center border-2 border-indigo-200">
                  <span className="text-gray-700 flex items-center gap-2 font-semibold">
                    <Play className="w-5 h-5 text-indigo-600" /> Audio Ready
                  </span>
                </div>
              ) : (
                <div className="text-center text-gray-600">
                  <Mic className="w-16 h-16 mx-auto mb-4 text-indigo-400 opacity-60 animate-pulse-soft" />
                  <p className="font-medium">Tap the microphone to start</p>
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-center gap-4">
              <button
                onClick={handleRecord}
                className={`px-8 py-4 rounded-2xl font-bold text-lg transition-all flex items-center gap-3 shadow-lg ${recording
                  ? "bg-gradient-to-r from-red-500 to-pink-500 text-white animate-pulse"
                  : "bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:scale-105"
                  }`}
              >
                {recording ? <><StopCircle className="w-6 h-6" /> Stop ({recordTime}s)</> : <><Mic className="w-6 h-6" /> Record Voice</>}
              </button>

              <div className="relative">
                <input type="file" onChange={handleFileChange} className="hidden" id="fileInput" />
                <label htmlFor="fileInput" className="cursor-pointer px-6 py-4 rounded-2xl bg-white/80 hover:bg-white border-2 border-indigo-200 text-indigo-700 font-semibold flex items-center gap-2 shadow-md">
                  <UploadCloud className="w-5 h-5" /> Upload Audio
                </label>
              </div>
            </div>

            {audioURL && (
              <div className="flex items-center justify-center gap-3 mt-6 fade-in">
                <button onClick={() => audioRef.current?.play()} className="p-4 rounded-xl bg-indigo-100 text-indigo-700"><Play className="w-5 h-5" /></button>
                <button onClick={handleSaveRecording} className="p-4 rounded-xl bg-purple-100 text-purple-700"><Save className="w-5 h-5" /></button>
                <button onClick={handleRetry} className="p-4 rounded-xl bg-pink-100 text-pink-700"><Trash className="w-5 h-5" /></button>
                <audio ref={audioRef} src={audioURL} controls hidden />
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={submitDisabled}
              className={`w-full mt-6 py-5 rounded-2xl font-bold text-xl flex items-center justify-center gap-3 shadow-xl ${submitDisabled ? "bg-gray-300 text-gray-500 cursor-not-allowed" : "bg-gradient-to-r from-pink-600 via-purple-600 to-indigo-600 text-white hover:scale-[1.02] animate-gradient"
                }`}
            >
              <Sparkles className="w-6 h-6" /> Analyze Emotion
            </button>

            {uploading && (
              <div className="w-full bg-indigo-100 rounded-full h-2 mt-4 overflow-hidden">
                <div className="bg-gradient-to-r from-indigo-600 to-pink-600 h-full" style={{ width: `${uploadProgress}%` }} />
              </div>
            )}
          </div>
        </div>

        {/* Timeline 1->2 */}
        {loading && <TimelineConnector active={currentStep >= 2} />}

        {/* ===== BLOCK 2: EMOTION DETECTION ===== */}
        {loading && (
          <div ref={block2Ref} className={`w-full block-container ${currentStep === 2 ? 'active' : ''} fade-in`}>
            <div className="flex items-center gap-4 mb-6">
              <StepBadge number={2} active={currentStep === 2} completed={currentStep > 2} />
              <h2 className="text-2xl font-bold text-gray-800">Detecting Emotion</h2>
            </div>
            <div className="glass-premium rounded-3xl p-8 shadow-xl flex flex-col items-center justify-center min-h-[300px]">
              <EmojiSwitcher emotion={DetectedEmotion} loading={loading} setisloading={setisloading} setemotion={setDetectedEmotion} />
              {DetectedEmotion && (
                <div className="mt-6 text-center fade-in">
                  <p className="text-sm uppercase tracking-widest text-gray-500 mb-2">Detected Emotion</p>
                  <p className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-pink-600">{DetectedEmotion}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Timeline 2->3 */}
        {currentStep >= 3 && <TimelineConnector active={currentStep >= 3} />}

        {/* ===== BLOCK 3: AI INSIGHTS ===== */}
        {currentStep >= 3 && (
          <div ref={block3Ref} className={`w-full block-container ${currentStep === 3 ? 'active' : ''} fade-in`}>
            <div className="flex items-center gap-4 mb-6">
              <StepBadge number={3} active={currentStep === 3} completed={false} />
              <h2 className="text-2xl font-bold text-gray-800">AI Insights</h2>
            </div>

            <div className="space-y-6">
              {(llmLoading || llmResponse) && (
                <div className="glass-premium rounded-3xl p-8 shadow-xl relative overflow-hidden">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-br from-indigo-600 to-pink-600 rounded-2xl shadow-lg animate-pulse-soft">
                      <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-800">Emotional Analysis</h3>
                      <p className="text-xs font-medium text-indigo-500 uppercase tracking-widest">AI Powered Insight</p>
                    </div>
                  </div>
                  {llmLoading && !llmResponse ? (
                    <div className="flex items-center gap-3 text-gray-500 py-4">
                      <div className="w-2.5 h-2.5 bg-indigo-600 rounded-full animate-bounce"></div>
                      <div className="w-2.5 h-2.5 bg-pink-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2.5 h-2.5 bg-purple-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <span className="ml-2 font-medium">Generating insights...</span>
                    </div>
                  ) : (
                    <div className="text-lg leading-relaxed text-gray-700 space-y-4">
                      {llmResponse.split('\n').map((line, index) => {
                        if (!line.trim()) return null;

                        // Function to render text with bold formatting
                        const renderWithBold = (text) => {
                          const parts = text.split(/\*\*(.*?)\*\*/g);
                          return parts.map((part, i) =>
                            i % 2 === 1 ? <strong key={i} className="font-bold text-gray-900">{part}</strong> : part
                          );
                        };

                        // Handle numbered lists (1. or 1))
                        if (/^\d+[\.\)]\s/.test(line.trim())) {
                          const num = line.match(/^\d+/)[0];
                          const text = line.replace(/^\d+[\.\)]\s*/, '');
                          return (
                            <div key={index} className="flex items-start gap-4 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-indigo-100">
                              <span className="w-8 h-8 flex-shrink-0 flex items-center justify-center bg-gradient-to-br from-indigo-500 to-purple-500 text-white rounded-full text-sm font-bold shadow-md">{num}</span>
                              <span className="flex-1">{renderWithBold(text)}</span>
                            </div>
                          );
                        }
                        // Handle bullet points
                        if (line.trim().startsWith('•') || line.trim().startsWith('-') || line.trim().startsWith('*')) {
                          return (
                            <div key={index} className="flex items-start gap-3 pl-2">
                              <span className="text-indigo-500 mt-1 text-xl">•</span>
                              <span>{renderWithBold(line.replace(/^[\s•\-\*]+/, ''))}</span>
                            </div>
                          );
                        }
                        // Regular paragraph
                        return <p key={index}>{renderWithBold(line)}</p>;
                      })}
                      {llmLoading && <span className="inline-block w-2 h-5 ml-1 bg-indigo-600 animate-pulse rounded-full"></span>}
                    </div>
                  )}
                </div>
              )}

              {(affirmationLoading || affirmation) && (
                <div className="glass-premium rounded-3xl p-6 shadow-xl relative overflow-hidden">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2.5 bg-gradient-to-br from-pink-500 to-rose-500 rounded-xl shadow-md">
                      <Heart className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold text-gray-800 text-lg">Daily Affirmation</span>
                  </div>
                  {affirmationLoading ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span className="text-sm font-medium">Crafting positivity...</span>
                    </div>
                  ) : (
                    <p className="text-xl font-serif italic text-gray-700 leading-relaxed pl-4 border-l-4 border-pink-300">"{affirmation}"</p>
                  )}
                </div>
              )}

              {(tipsLoading || wellnessTips) && (
                <div className="glass-premium rounded-3xl p-6 shadow-xl relative overflow-hidden">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2.5 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-xl shadow-md">
                      <Lightbulb className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold text-gray-800 text-lg">Wellness Tips</span>
                  </div>
                  {tipsLoading ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span className="text-sm font-medium">Curating tips...</span>
                    </div>
                  ) : (
                    <div className="text-gray-700 leading-relaxed whitespace-pre-line font-medium">{wellnessTips}</div>
                  )}
                </div>
              )}

              <button onClick={handleRetry} className="w-full py-4 rounded-2xl font-bold text-lg bg-white/80 hover:bg-white border-2 border-indigo-200 text-indigo-700 flex items-center justify-center gap-3 shadow-md">
                <RefreshCw className="w-5 h-5" /> Try Another Recording
              </button>
            </div>
          </div>
        )}

        <div className="mt-12 mb-8 text-center text-gray-600">
          <p className="text-sm font-medium">Emotion Detection</p>
        </div>
      </div>
    </div>
  );
}
