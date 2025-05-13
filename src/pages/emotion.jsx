import { useState, useRef } from "react";
import { UploadCloud, Mic, Send, Play, Trash, Save, MenuIcon, Cross, LucideDoorOpen } from "lucide-react";
import axios from "axios";
import "./emotion.css";
import EmojiSwitcher from "../components/ui/emoji";
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
  const [loading, setisloading] = useState(false);
  const [emotiondetected, setisemotiondetected] = useState(false);

  const mediaRecorderRef = useRef(null);
  const timerRef = useRef(null);
  const streamRef = useRef(null);
  const audioRef = useRef(null);
 
  // Handle file upload
  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    if (file && (file.type === "audio/wav" || file.type === "audio/mpeg")) {
      setSelectedFile(file);
      setAudioBlob(null);
      setAudioURL(null);
      setUploading(true);
      setSubmitDisabled(true);

      // Simulate file upload progress
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

  // Start/Stop recording
  const handleRecord = async () => {
    if (!recording) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });

      mediaRecorderRef.current = mediaRecorder;

      let audioChunks = [];
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/mpeg" });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioBlob(audioBlob);
        setAudioURL(audioUrl);
        setRecordTime(0);
        clearInterval(timerRef.current);
        setSubmitDisabled(false);

        // Release media stream tracks
        streamRef.current?.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setRecording(true);

      // Start Timer
      timerRef.current = setInterval(() => {
        setRecordTime((prevTime) => prevTime + 1);
      }, 1000);
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
          const audioBuffer = await audioContext.decodeAudioData(
            fileReader.result
          );
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
    const format = 1; // PCM
    const bitDepth = 16;
    let offset = 0;

    const samples = audioBuffer.getChannelData(0);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(view, offset, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numOfChan, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numOfChan * (bitDepth / 8), true);
    view.setUint16(32, numOfChan * (bitDepth / 8), true);
    view.setUint16(34, bitDepth, true);
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

  // Save recorded audio as a selected file
  const handleSaveRecording = async () => {
    if (audioBlob) {
      try {
        const wavBlob = await convertWebMtoWAV(audioBlob);
        const newFile = new File([wavBlob], "recorded_audio.wav", {
          type: "audio/wav",
        });
        setSelectedFile(newFile);
        alert("Recording saved as WAV file!");
      } catch (error) {
        console.error("Error converting WebM to WAV:", error);
        alert("Conversion failed!");
      }
    }
  };

  // Submit file or recorded audio to API
  const handleSubmit = async () => {
    setisloading(true);
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
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setDetectedEmotion(
        response.data.predicted_emotion.slice(4).toUpperCase()
      );
      
      setisemotiondetected(true);
      alert(
        "Detected Emotion: " +
          response.data.predicted_emotion.slice(4).toUpperCase()
      );
    } catch (error) {
      console.error("Error:", error);
      alert("Error detecting emotion. Please try again.");
    } finally {
    }
  };

  // Retry Recording
  const handleRetry = () => {
    setAudioBlob(null);
    setAudioURL(null);
    setSelectedFile(null);
    setSubmitDisabled(true);
  };

  return (
    <div className="h-[100vh] flex flex-col  justify-top bg-gradient-to-br from-indigo-600 via-purple-700 to-pink-600 text-white pb-[1%] pt-[1%]">
      {/* Header */}
    
     <div className="flex flex-col items-center">
     <h1 className="text-3xl md:text-5xl lg:text-6xl  text-center mb-[3%] tracking-wide  bg-clip-text text-transparent animate-text-glow mt-[8%]">
        emotion detection using speech
      </h1>

      {/* Full-Screen Card */}
      <div className="w-[80%] md:w-[70%] lg:w-[60%] max-w-4xl h-fit max-h-[90vh] bg-white p-[3%] shadow-2xl rounded-3xl text-black flex flex-col justify-center mb-[4%]">
        {loading ? (
          <EmojiSwitcher
            emotion={DetectedEmotion}
            loading={loading}
            setisloading={setisloading}
            setemotion={setDetectedEmotion}
          />
        ) : (
          <div className="flex flex-col items-center gap-4 w-full">
            {/* File Upload */}
            <label
              htmlFor="fileInput"
              className="cursor-pointer flex items-center justify-center gap-3 w-[70%] bg-gray-200 hover:bg-gray-300 text-gray-700 font-semibold py-4 rounded-xl shadow transition duration-300"
            >
              <UploadCloud className="w-6 h-6" /> upload WAV/MP3
            </label>
            <input
              type="file"
              onChange={handleFileChange}
              className="hidden"
              id="fileInput"
            />

            {/* Progress Bar */}
            {uploading && (
              <div className="w-[70%] bg-gray-300 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            )}

            {/* Record Button with Timer */}
            <button
              onClick={handleRecord}
              className={`w-[70%] flex items-center justify-center gap-3 py-4 rounded-xl font-semibold transition-all duration-300 ${
                recording
                  ? "bg-red-500 hover:bg-red-700"
                  : "bg-blue-500 hover:bg-blue-700"
              }`}
            >
              <Mic className="w-6 h-6" />{" "}
              {recording ? `Stop Recording (${recordTime}s)` : "record audio"}
            </button>

            {/* Playback & Save Options */}
            {audioURL && (
              <div className="flex items-center gap-4 w-[70%]">
                <button
                  onClick={() => audioRef.current?.play()}
                  className="bg-purple-500 hover:bg-purple-700 w-1/3 py-3 rounded-xl"
                >
                  <Play className="w-5 h-5" /> Listen
                </button>
                <button
                  onClick={handleSaveRecording}
                  className="bg-yellow-500 hover:bg-yellow-700 w-1/3 py-3 rounded-xl"
                >
                  <Save className="w-5 h-5" /> Save
                </button>
                <button
                  onClick={handleRetry}
                  className="bg-red-500 hover:bg-red-700 w-1/3 py-3 rounded-xl"
                >
                  <Trash className="w-5 h-5" /> Retry
                </button>
                <audio ref={audioRef} src={audioURL} controls hidden />
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={submitDisabled}
              className={`w-[70%] flex items-center justify-center gap-3 font-semibold py-4 rounded-xl transition-all duration-300 ${
                submitDisabled
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-green-500 hover:bg-green-700"
              }`}
            >
              <Send className="w-6 h-6" /> detect emotion
            </button>
          </div>
        )}
      </div>
     </div>
    </div>
  );
}
