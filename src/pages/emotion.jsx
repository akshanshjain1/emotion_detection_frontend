import { useEffect, useRef, useState } from "react";
import {
  LoaderCircle,
  Mic,
  Sparkles,
  Square,
  Trash2,
  UploadCloud,
} from "lucide-react";
import "./emotion.css";
import AudioVisualizer from "../components/ui/AudioVisualizer";
import { generateGroqStream } from "../utils/api";
import { predictEmotionFromAudioFile } from "../utils/humeEmotionApi";

function formatLabel(emotion) {
  return emotion.charAt(0).toUpperCase() + emotion.slice(1);
}

function getEmotionEmoji(emotion) {
  const emojiMap = {
    neutral: "😐",
    calm: "😌",
    happy: "😄",
    sad: "😢",
    angry: "😠",
    fearful: "😨",
    disgust: "🤢",
    surprise: "😲",
  };

  return emojiMap[emotion] ?? "🙂";
}

function getSupportedRecorderMimeType() {
  const preferredMimeTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

  for (const mimeType of preferredMimeTypes) {
    if (window.MediaRecorder?.isTypeSupported?.(mimeType)) {
      return mimeType;
    }
  }

  return "";
}

function getFileExtension(mimeType) {
  if (mimeType.includes("mp4")) {
    return "mp4";
  }

  if (mimeType.includes("webm")) {
    return "webm";
  }

  return "webm";
}

async function convertRecordedBlobToWavFile(audioBlob) {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();

  try {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const wavBlob = encodeAudioBufferAsWav(audioBuffer);

    return new File([wavBlob], `voice-recording-${Date.now()}.wav`, {
      type: "audio/wav",
    });
  } finally {
    if (audioContext.state !== "closed") {
      await audioContext.close();
    }
  }
}

function encodeAudioBufferAsWav(audioBuffer) {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const sampleCount = audioBuffer.length;
  const monoSamples = new Float32Array(sampleCount);

  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const channelData = audioBuffer.getChannelData(channelIndex);

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      monoSamples[sampleIndex] += channelData[sampleIndex] / channelCount;
    }
  }

  const wavBuffer = new ArrayBuffer(44 + monoSamples.length * 2);
  const view = new DataView(wavBuffer);

  writeAsciiString(view, 0, "RIFF");
  view.setUint32(4, 36 + monoSamples.length * 2, true);
  writeAsciiString(view, 8, "WAVE");
  writeAsciiString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAsciiString(view, 36, "data");
  view.setUint32(40, monoSamples.length * 2, true);

  let offset = 44;

  for (let sampleIndex = 0; sampleIndex < monoSamples.length; sampleIndex += 1) {
    const sample = Math.max(-1, Math.min(1, monoSamples[sampleIndex]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([wavBuffer], { type: "audio/wav" });
}

function writeAsciiString(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

export default function EmotionDetection() {
  const groqApiKey = import.meta.env.VITE_GROQ_API_KEY?.trim();

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [aiResponse, setAiResponse] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [isGeneratingResponse, setIsGeneratingResponse] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [responseError, setResponseError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioStream, setAudioStream] = useState(null);
  const [audioSourceLabel, setAudioSourceLabel] = useState("");

  const mediaRecorderRef = useRef(null);
  const timerRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      stopActiveStream();
      clearRecordingTimer();
    };
  }, []);

  const updateSelectedAudio = (file, sourceLabel) => {
    setSelectedFile(file);
    setAudioSourceLabel(sourceLabel);
    setPrediction(null);
    setAiResponse("");
    setErrorMessage("");
    setResponseError("");

    setPreviewUrl((currentPreviewUrl) => {
      if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
      }

      return file ? URL.createObjectURL(file) : "";
    });
  };

  const clearAllState = () => {
    updateSelectedAudio(null, "");
    setPrediction(null);
    setAiResponse("");
    setErrorMessage("");
    setResponseError("");
    setIsPredicting(false);
    setIsGeneratingResponse(false);
    setRecordingSeconds(0);

    if (isRecording && mediaRecorderRef.current?.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    setIsRecording(false);
    clearRecordingTimer();
    stopActiveStream();
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null;

    if (!file) {
      return;
    }

    if (isRecording) {
      stopRecording();
    }

    updateSelectedAudio(file, "Uploaded Audio");
  };

  const handleRecordToggle = async () => {
    if (isRecording) {
      stopRecording();
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage("Audio recording is not supported in this browser.");
      return;
    }

    if (!window.MediaRecorder) {
      setErrorMessage("MediaRecorder is not available in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedRecorderMimeType();
      const mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      const audioChunks = [];

      streamRef.current = stream;
      mediaRecorderRef.current = mediaRecorder;
      setAudioStream(stream);
      setIsRecording(true);
      setRecordingSeconds(0);
      setPrediction(null);
      setAiResponse("");
      setErrorMessage("");
      setResponseError("");
      setAudioSourceLabel("Live Recording");

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        clearRecordingTimer();
        setIsRecording(false);

        const recordedMimeType = mediaRecorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(audioChunks, { type: recordedMimeType });

        if (blob.size === 0) {
          setErrorMessage("No audio was captured. Please try recording again.");
          stopActiveStream();
          return;
        }

        try {
          const recordedFile = await convertRecordedBlobToWavFile(blob);
          updateSelectedAudio(recordedFile, "Recorded Audio");
        } catch {
          const extension = getFileExtension(recordedMimeType);
          const recordedFile = new File(
            [blob],
            `voice-recording-${Date.now()}.${extension}`,
            { type: recordedMimeType }
          );

          updateSelectedAudio(recordedFile, "Recorded Audio");
          setErrorMessage("Recording saved, but format conversion was unavailable. If analysis fails, try uploading a WAV, MP3, or MP4 file.");
        }

        stopActiveStream();
      };

      mediaRecorder.start();
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((currentSeconds) => currentSeconds + 1);
      }, 1000);
    } catch {
      setErrorMessage("Microphone access was denied or unavailable.");
      stopActiveStream();
      setIsRecording(false);
      clearRecordingTimer();
    }
  };

  const stopRecording = () => {
    clearRecordingTimer();

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    } else {
      setIsRecording(false);
      stopActiveStream();
    }
  };

  const handlePredictEmotion = async () => {
    if (!selectedFile) {
      setErrorMessage("Upload or record an audio clip before analyzing it.");
      return;
    }

    setIsPredicting(true);
    setPrediction(null);
    setAiResponse("");
    setErrorMessage("");
    setResponseError("");

    try {
      const result = await predictEmotionFromAudioFile(selectedFile);
      console.table(result.rawHumeTopEmotions);
      setPrediction(result);

      if (!groqApiKey) {
        setResponseError("AI feedback is unavailable right now.");
        return;
      }

      try {
        setIsGeneratingResponse(true);
        await generateGroqStream(
          result.predictedEmotion,
          groqApiKey,
          (chunk) => {
            setAiResponse((currentResponse) => currentResponse + chunk);
          },
          "comprehensive"
        );
      } catch {
        setResponseError("The emotion was detected, but the AI response could not be generated.");
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Emotion analysis failed.");
    } finally {
      setIsPredicting(false);
      setIsGeneratingResponse(false);
    }
  };

  const isBusy = isPredicting || isGeneratingResponse;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(191,219,254,0.95),_rgba(224,231,255,0.94)_28%,_rgba(250,245,255,1)_72%)] px-4 py-12">
      <div className="mx-auto max-w-5xl">
        <div className="rounded-[2rem] border border-white/60 bg-white/70 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.12)] backdrop-blur">
          <div className="mb-10 text-center">
            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.35em] text-indigo-600">
              Emotion Analysis
            </p>
            <h1 className="text-4xl font-black tracking-tight text-slate-900 md:text-5xl">
              Voice Emotion Prediction
            </h1>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
            <section className="rounded-[1.75rem] border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-5 flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold text-slate-900">Record or Upload</h2>
                  <p className="text-sm text-slate-500">
                    Supported: MP3, WAV, WEBM, OGG, MP4, M4A
                  </p>
                </div>
                {audioSourceLabel && (
                  <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-indigo-700">
                    {audioSourceLabel}
                  </span>
                )}
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label
                  htmlFor="audio-file"
                  className={`flex cursor-pointer items-center justify-center gap-3 rounded-2xl border px-5 py-4 text-base font-semibold transition ${
                    isRecording || isBusy
                      ? "pointer-events-none border-slate-200 bg-slate-100 text-slate-400"
                      : "border-indigo-200 bg-indigo-50 text-indigo-700 hover:border-indigo-400 hover:bg-indigo-100"
                  }`}
                >
                  <UploadCloud className="h-5 w-5" />
                  Upload Audio
                </label>
                <input
                  id="audio-file"
                  type="file"
                  accept=".mp3,.wav,.webm,.ogg,.mp4,.m4a,audio/mpeg,audio/mp3,audio/wav,audio/x-wav,audio/webm,audio/ogg,audio/mp4,audio/m4a"
                  className="hidden"
                  onChange={handleFileChange}
                />

                <button
                  type="button"
                  onClick={handleRecordToggle}
                  disabled={isBusy}
                  className={`flex items-center justify-center gap-3 rounded-2xl px-5 py-4 text-base font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-70 ${
                    isRecording
                      ? "bg-gradient-to-r from-rose-500 to-red-500 hover:from-rose-600 hover:to-red-600"
                      : "bg-gradient-to-r from-slate-900 to-indigo-900 hover:from-slate-800 hover:to-indigo-800"
                  }`}
                >
                  {isRecording ? (
                    <>
                      <Square className="h-5 w-5" />
                      Stop Recording
                    </>
                  ) : (
                    <>
                      <Mic className="h-5 w-5" />
                      Record Live Audio
                    </>
                  )}
                </button>
              </div>

              <div className="mt-6 rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5">
                {isRecording ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between rounded-2xl bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">
                      <span>Recording in progress</span>
                      <span>{recordingSeconds}s</span>
                    </div>
                    <AudioVisualizer stream={audioStream} isRecording={isRecording} />
                  </div>
                ) : previewUrl ? (
                  <div className="space-y-4">
                    <div className="rounded-2xl bg-white p-4 shadow-sm">
                      <audio controls src={previewUrl} className="w-full" />
                    </div>
                    {selectedFile && (
                      <div className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
                        <p>
                          <span className="font-semibold text-slate-800">File:</span>{" "}
                          {selectedFile.name}
                        </p>
                        <p>
                          <span className="font-semibold text-slate-800">Type:</span>{" "}
                          {selectedFile.type || "Unknown"}
                        </p>
                        <p>
                          <span className="font-semibold text-slate-800">Size:</span>{" "}
                          {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex min-h-52 flex-col items-center justify-center rounded-[1.25rem] border-2 border-dashed border-slate-200 bg-white px-6 text-center">
                    <Mic className="mb-4 h-12 w-12 text-indigo-400" />
                    <p className="text-lg font-semibold text-slate-800">
                      Start with a voice sample
                    </p>
                    <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
                      Record a short clip or upload an existing audio file to begin the analysis.
                    </p>
                  </div>
                )}
              </div>

              <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={handlePredictEmotion}
                  disabled={!selectedFile || isBusy || isRecording}
                  className="inline-flex flex-1 items-center justify-center gap-3 rounded-2xl bg-slate-900 px-5 py-4 text-base font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                >
                  {isPredicting ? (
                    <>
                      <LoaderCircle className="h-5 w-5 animate-spin" />
                      Analyzing Emotion...
                    </>
                  ) : isGeneratingResponse ? (
                    <>
                      <LoaderCircle className="h-5 w-5 animate-spin" />
                      Writing Response...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-5 w-5" />
                      Analyze Emotion
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={clearAllState}
                  disabled={isBusy || isRecording}
                  className="inline-flex items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-4 text-base font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Trash2 className="h-5 w-5" />
                  Clear
                </button>
              </div>

              {errorMessage && (
                <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {errorMessage}
                </div>
              )}
            </section>

            <section className="rounded-[1.75rem] border border-slate-200 bg-slate-950 p-6 text-slate-100 shadow-sm">
              <p className="text-sm font-semibold uppercase tracking-[0.3em] text-indigo-300">
                Prediction
              </p>

              {prediction ? (
                <div className="mt-6 space-y-6">
                  <div className="rounded-[1.5rem] bg-white/10 p-5">
                    <p className="text-sm text-slate-400">Predicted Emotion</p>
                    <div className="mt-3 flex items-center gap-4">
                      <span className="text-5xl leading-none">
                        {getEmotionEmoji(prediction.predictedEmotion)}
                      </span>
                      <p className="text-4xl font-black tracking-tight text-white">
                        {formatLabel(prediction.predictedEmotion)}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-8 rounded-[1.5rem] border border-dashed border-white/10 bg-white/5 p-6 text-sm leading-7 text-slate-300">
                  Your emotion prediction will appear here after analysis.
                </div>
              )}
            </section>
          </div>

          {(prediction || isGeneratingResponse || aiResponse || responseError) && (
            <section className="mt-6 rounded-[1.75rem] border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-5 flex items-center gap-3">
                <div className="rounded-2xl bg-fuchsia-100 p-3 text-fuchsia-700">
                  <Sparkles className="h-6 w-6" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-900">AI Response</h2>
                  <p className="text-sm text-slate-500">
                    Personalized guidance based on the detected emotion
                  </p>
                </div>
              </div>

              {isGeneratingResponse && !aiResponse ? (
                <div className="flex items-center gap-3 rounded-2xl bg-slate-50 px-4 py-4 text-slate-600">
                  <LoaderCircle className="h-5 w-5 animate-spin text-indigo-600" />
                  Generating your response...
                </div>
              ) : aiResponse ? (
                <div className="rounded-[1.5rem] bg-[linear-gradient(135deg,rgba(238,242,255,0.95),rgba(250,245,255,0.95))] p-5 text-base leading-8 text-slate-700 whitespace-pre-line">
                  {aiResponse}
                </div>
              ) : null}

              {responseError && (
                <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                  {responseError}
                </div>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  );

  function clearRecordingTimer() {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function stopActiveStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setAudioStream(null);
  }
}
