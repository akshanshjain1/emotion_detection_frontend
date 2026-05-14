const HUME_BATCH_API_BASE_URL = "https://api.hume.ai/v0/batch/jobs";

export const ALLOWED_EMOTIONS = Object.freeze([
  "neutral",
  "calm",
  "happy",
  "sad",
  "angry",
  "fearful",
  "disgust",
  "surprise",
]);

export const SUPPORTED_AUDIO_TYPES = Object.freeze([
  "audio/mpeg",
  "audio/mp3",
  "audio/wav",
  "audio/x-wav",
  "audio/webm",
  "audio/ogg",
  "audio/mp4",
  "audio/m4a",
  "audio/x-m4a",
]);

const SUPPORTED_AUDIO_EXTENSIONS = new Set(["mp3", "wav", "webm", "ogg", "mp4", "m4a"]);
const SUPPORTED_AUDIO_TYPE_SET = new Set(SUPPORTED_AUDIO_TYPES);
const SUPPORTED_TRANSCRIPTION_LANGUAGES = new Set([
  "zh",
  "da",
  "nl",
  "en",
  "en-AU",
  "en-IN",
  "en-NZ",
  "en-GB",
  "fr",
  "fr-CA",
  "de",
  "hi",
  "hi-Latn",
  "id",
  "it",
  "ja",
  "ko",
  "no",
  "pl",
  "pt",
  "pt-BR",
  "pt-PT",
  "ru",
  "es",
  "es-419",
  "sv",
  "ta",
  "tr",
  "uk",
]);
const JOB_TIMEOUT_MS = 2 * 60 * 1000;
const POLL_INTERVAL_MS = 2500;
const DEFAULT_MAPPED_EMOTION = "neutral";
const ANALYSIS_SAMPLE_RATE = 16000;
const SILENCE_THRESHOLD = 0.015;
const SILENCE_PADDING_SECONDS = 0.12;
const TARGET_PEAK_AMPLITUDE = 0.92;

// Demo/prototype note:
// Hume's official docs say Expression Measurement is being sunset.
// Playground job creation ends on May 14, 2026, and API access/result downloads continue until June 14, 2026.
// This browser-direct integration is okay for a prototype, but it is not a secure or long-term production setup.
const EMOTION_MAP = Object.freeze({
  Anger: "angry",
  Annoyance: "angry",
  Frustration: "angry",

  Contempt: "disgust",
  Disgust: "disgust",

  Fear: "fearful",
  Horror: "fearful",
  Distress: "fearful",
  Anxiety: "fearful",
  Worry: "fearful",

  Sadness: "sad",
  Guilt: "sad",
  Pain: "sad",
  Tiredness: "sad",

  Calmness: "calm",
  Contentment: "calm",
  Relief: "calm",

  Amusement: "happy",
  Joy: "happy",
  Excitement: "happy",
  Pride: "happy",
  Desire: "happy",

  Surprise: "surprise",
  "Surprise (positive)": "surprise",
  "Surprise (negative)": "surprise",
  Awe: "surprise",
  Realization: "surprise",

  Neutral: "neutral",
  Boredom: "neutral",
  Awkwardness: "neutral",
  Confusion: "neutral",
  Concentration: "neutral",
  Determination: "neutral",
  Interest: "neutral",
});

export async function predictEmotionFromAudioFile(audioFile, options = {}) {
  const apiKey = getHumeApiKey();
  validateAudioFile(audioFile);
  const preparedAudioFile = await prepareAudioFileForAnalysis(audioFile);
  const transcriptionLanguageCandidates = buildTranscriptionLanguageCandidates(options.language);
  let lastError = null;

  for (
    let languageIndex = 0;
    languageIndex < transcriptionLanguageCandidates.length;
    languageIndex += 1
  ) {
    const transcriptionLanguage = transcriptionLanguageCandidates[languageIndex];
    const analysisModes = ["utterance", "window"];

    for (let attemptIndex = 0; attemptIndex < analysisModes.length; attemptIndex += 1) {
      const analysisMode = analysisModes[attemptIndex];

      try {
        return await runAnalysisAttempt(
          preparedAudioFile,
          apiKey,
          analysisMode,
          transcriptionLanguage
        );
      } catch (error) {
        lastError = error;

        const isLastLanguageAttempt = languageIndex === transcriptionLanguageCandidates.length - 1;
        const isLastModeAttempt = attemptIndex === analysisModes.length - 1;

        if (!isRetryableEmptyAnalysis(error) || (isLastLanguageAttempt && isLastModeAttempt)) {
          throw error;
        }
      }
    }
  }

  throw lastError ?? new Error("Emotion analysis failed.");
}

async function prepareAudioFileForAnalysis(audioFile) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;

  if (!AudioContextClass) {
    return audioFile;
  }

  const audioContext = new AudioContextClass();

  try {
    const arrayBuffer = await audioFile.arrayBuffer();
    const decodedAudioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const monoSamples = downmixToMono(decodedAudioBuffer);
    const trimmedSamples = trimSilence(monoSamples, decodedAudioBuffer.sampleRate);
    const normalizedSamples = normalizeSamples(trimmedSamples);
    const resampledSamples = resampleLinear(
      normalizedSamples,
      decodedAudioBuffer.sampleRate,
      ANALYSIS_SAMPLE_RATE
    );
    const wavBlob = createWavBlob(resampledSamples, ANALYSIS_SAMPLE_RATE);

    return new File([wavBlob], replaceFileExtension(audioFile.name, "wav"), {
      type: "audio/wav",
    });
  } catch {
    return audioFile;
  } finally {
    if (audioContext.state !== "closed") {
      await audioContext.close();
    }
  }
}

function getHumeApiKey() {
  const apiKey = import.meta.env.VITE_HUME_API_KEY?.trim();

  if (!apiKey) {
    throw new Error("Missing speech analysis API key. Set VITE_HUME_API_KEY in your environment.");
  }

  return apiKey;
}

function validateAudioFile(audioFile) {
  if (!audioFile) {
    throw new Error("No audio file selected.");
  }

  if (!(audioFile instanceof Blob) || typeof audioFile.name !== "string") {
    throw new Error("Expected a browser File object for audio prediction.");
  }

  const fileType = audioFile.type?.toLowerCase() ?? "";
  const fileExtension = audioFile.name.split(".").pop()?.toLowerCase() ?? "";
  const isSupportedType = SUPPORTED_AUDIO_TYPE_SET.has(fileType);
  const isSupportedExtension = SUPPORTED_AUDIO_EXTENSIONS.has(fileExtension);

  if (!isSupportedType && !isSupportedExtension) {
    throw new Error(
      `Unsupported file type. Supported types: ${SUPPORTED_AUDIO_TYPES.join(", ")}`
    );
  }
}

async function runAnalysisAttempt(audioFile, apiKey, analysisMode, transcriptionLanguage) {
  const jobConfig = createBatchConfig(analysisMode, transcriptionLanguage);
  const jobId = await startInferenceJob(audioFile, apiKey, jobConfig);
  const jobDetails = await waitForJobCompletion(jobId, apiKey);
  const predictionPayload = await fetchJobPredictions(jobId, apiKey);
  const utteranceEmotionScores = extractProsodyEmotionScores(predictionPayload);

  if (shouldTreatAsEmptyAnalysis(jobDetails, utteranceEmotionScores)) {
    throw createEmptyAnalysisError(
      audioFile,
      predictionPayload,
      jobDetails,
      analysisMode,
      transcriptionLanguage
    );
  }

  const topEmotions = aggregateMappedEmotionScores(utteranceEmotionScores);
  const rawHumeTopEmotions = aggregateRawEmotionScores(utteranceEmotionScores).slice(0, 5);
  const predictedEmotion = topEmotions[0]?.emotion ?? DEFAULT_MAPPED_EMOTION;
  const confidence = topEmotions[0]?.score ?? 0;

  return {
    predictedEmotion,
    confidence,
    datasetCompatible: true,
    allowedEmotions: [...ALLOWED_EMOTIONS],
    topEmotions,
    rawHumeTopEmotions,
    note: "Prediction is mapped to RAVDESS/CREMA-compatible emotion classes.",
  };
}

function createBatchConfig(analysisMode, transcriptionLanguage) {
  const prosodyConfig =
    analysisMode === "window"
      ? {
          window: {
            length: 2,
            step: 1,
          },
        }
      : {
          granularity: "utterance",
        };
  const transcriptionConfig = getTranscriptionConfig(transcriptionLanguage);

  return {
    models: {
      prosody: prosodyConfig,
    },
    notify: false,
    ...(transcriptionConfig ? { transcription: transcriptionConfig } : {}),
  };
}

function getTranscriptionConfig(transcriptionLanguage) {
  if (!transcriptionLanguage) {
    return null;
  }

  return {
    language: transcriptionLanguage,
  };
}

function buildTranscriptionLanguageCandidates(explicitLanguage) {
  const candidates = [];
  const normalizedExplicitLanguage = normalizeRequestedLanguage(explicitLanguage);

  if (normalizedExplicitLanguage) {
    candidates.push(normalizedExplicitLanguage);
  } else {
    candidates.push(null);

    const browserLanguage = typeof navigator !== "undefined" ? navigator.language : "";
    const normalizedBrowserLanguage = normalizeRequestedLanguage(browserLanguage);

    if (normalizedBrowserLanguage) {
      candidates.push(normalizedBrowserLanguage);
    }

    const regionLikelyIndia = isLikelyIndiaLocale();

    if (regionLikelyIndia) {
      candidates.push("hi-Latn");
      candidates.push("hi");
      candidates.push("en-IN");
    }

    candidates.push("en");
  }

  return dedupePreservingOrder(candidates);
}

function normalizeRequestedLanguage(languageTag) {
  if (!languageTag || typeof languageTag !== "string" || languageTag === "auto") {
    return null;
  }

  if (SUPPORTED_TRANSCRIPTION_LANGUAGES.has(languageTag)) {
    return languageTag;
  }

  const baseLanguage = languageTag.split("-")[0];
  return SUPPORTED_TRANSCRIPTION_LANGUAGES.has(baseLanguage) ? baseLanguage : null;
}

function isLikelyIndiaLocale() {
  if (typeof navigator !== "undefined") {
    const languageList = Array.isArray(navigator.languages) ? navigator.languages : [];

    if (languageList.some((language) => typeof language === "string" && language.includes("-IN"))) {
      return true;
    }

    if (typeof navigator.language === "string" && navigator.language.includes("-IN")) {
      return true;
    }
  }

  if (typeof Intl !== "undefined" && Intl.DateTimeFormat) {
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone ?? "";
    return timeZone === "Asia/Kolkata" || timeZone === "Asia/Calcutta";
  }

  return false;
}

async function startInferenceJob(audioFile, apiKey, jobConfig) {
  const primaryAttempt = await submitInferenceJob(audioFile, apiKey, "file[]", jobConfig);

  if (primaryAttempt.ok) {
    return extractJobIdFromResponse(primaryAttempt);
  }

  const primaryErrorText = await primaryAttempt.text();

  if (shouldRetryWithSingularFileField(primaryAttempt.status, primaryErrorText)) {
    const fallbackAttempt = await submitInferenceJob(audioFile, apiKey, "file", jobConfig);

    if (fallbackAttempt.ok) {
      return extractJobIdFromResponse(fallbackAttempt);
    }

    throw buildApiErrorFromText(
      "Failed to start audio analysis",
      fallbackAttempt.status,
      fallbackAttempt.statusText,
      await fallbackAttempt.text()
    );
  }

  throw buildApiErrorFromText(
    "Failed to start audio analysis",
    primaryAttempt.status,
    primaryAttempt.statusText,
    primaryErrorText
  );
}

async function submitInferenceJob(audioFile, apiKey, fileFieldName, jobConfig) {
  const formData = new FormData();
  formData.append(fileFieldName, audioFile, audioFile.name);
  formData.append("json", JSON.stringify(jobConfig));

  return fetch(HUME_BATCH_API_BASE_URL, {
    method: "POST",
    headers: {
      "X-Hume-Api-Key": apiKey,
    },
    body: formData,
  });
}

async function extractJobIdFromResponse(response) {
  const payload = await response.json();
  const jobId = payload?.job_id ?? payload?.jobId;

  if (!jobId) {
    throw new Error("Audio analysis started, but the response did not include a job ID.");
  }

  return jobId;
}

async function waitForJobCompletion(jobId, apiKey) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < JOB_TIMEOUT_MS) {
    const response = await fetch(`${HUME_BATCH_API_BASE_URL}/${jobId}`, {
      method: "GET",
      headers: {
        "X-Hume-Api-Key": apiKey,
      },
    });

    if (!response.ok) {
      throw await buildApiError("Failed to check analysis status", response);
    }

    const jobStatusPayload = await response.json();
    const status = normalizeStatus(
      jobStatusPayload?.state?.status ??
        jobStatusPayload?.status ??
        jobStatusPayload?.job_status ??
        jobStatusPayload?.jobStatus
    );

    if (status === "COMPLETED") {
      return jobStatusPayload;
    }

    if (status === "FAILED") {
      throw new Error("Audio analysis failed for this recording.");
    }

    await delay(POLL_INTERVAL_MS);
  }

  throw new Error("Audio analysis timed out after 2 minutes.");
}

async function fetchJobPredictions(jobId, apiKey) {
  const response = await fetch(`${HUME_BATCH_API_BASE_URL}/${jobId}/predictions`, {
    method: "GET",
    headers: {
      "X-Hume-Api-Key": apiKey,
    },
  });

  if (!response.ok) {
    throw await buildApiError("Failed to fetch analysis results", response);
  }

  return response.json();
}

function extractProsodyEmotionScores(predictionPayload) {
  const utteranceEmotionScores = [];
  const sourceItems = Array.isArray(predictionPayload)
    ? predictionPayload
    : predictionPayload
      ? [predictionPayload]
      : [];

  for (const sourceItem of sourceItems) {
    const filePredictions = extractFilePredictions(sourceItem);

    for (const filePrediction of filePredictions) {
      const prosodyModel = filePrediction?.models?.prosody;

      if (!prosodyModel) {
        continue;
      }

      const directPredictions = toArray(prosodyModel.predictions);
      const groupedPredictions = toArray(
        prosodyModel.grouped_predictions ?? prosodyModel.groupedPredictions
      );
      const allPredictions = [...directPredictions];

      for (const groupedPrediction of groupedPredictions) {
        allPredictions.push(...toArray(groupedPrediction?.predictions));
      }

      for (const prediction of allPredictions) {
        const emotions = toArray(prediction?.emotions)
          .map(normalizeEmotionScore)
          .filter(Boolean);

        if (emotions.length > 0) {
          utteranceEmotionScores.push(emotions);
        }
      }
    }
  }

  return utteranceEmotionScores;
}

function extractFilePredictions(sourceItem) {
  if (!sourceItem || typeof sourceItem !== "object") {
    return [];
  }

  const directPredictions = toArray(sourceItem?.predictions);
  const resultPredictions = toArray(sourceItem?.results?.predictions);
  const responsePredictions = toArray(sourceItem?.response?.predictions);

  if (resultPredictions.length > 0) {
    return resultPredictions;
  }

  if (responsePredictions.length > 0) {
    return responsePredictions;
  }

  if (directPredictions.length > 0) {
    return directPredictions;
  }

  if (sourceItem?.models?.prosody) {
    return [sourceItem];
  }

  return [];
}

function normalizeEmotionScore(emotionEntry) {
  const emotion = emotionEntry?.name ?? emotionEntry?.emotion;
  const score = Number(emotionEntry?.score);

  if (!emotion || Number.isNaN(score)) {
    return null;
  }

  return {
    emotion,
    score,
  };
}

function aggregateMappedEmotionScores(utteranceEmotionScores) {
  const buckets = ALLOWED_EMOTIONS.reduce((accumulator, emotion) => {
    accumulator[emotion] = [];
    return accumulator;
  }, {});

  for (const utterance of utteranceEmotionScores) {
    for (const { emotion, score } of utterance) {
      const mappedEmotion = EMOTION_MAP[emotion] ?? DEFAULT_MAPPED_EMOTION;
      buckets[mappedEmotion].push(score);
    }
  }

  return ALLOWED_EMOTIONS.map((emotion) => ({
    emotion,
    score: roundScore(average(buckets[emotion])),
  })).sort(sortByScoreDescending);
}

function aggregateRawEmotionScores(utteranceEmotionScores) {
  const rawBuckets = new Map();

  for (const utterance of utteranceEmotionScores) {
    for (const { emotion, score } of utterance) {
      const existingScores = rawBuckets.get(emotion) ?? [];
      existingScores.push(score);
      rawBuckets.set(emotion, existingScores);
    }
  }

  return [...rawBuckets.entries()]
    .map(([emotion, scores]) => ({
      emotion,
      score: roundScore(average(scores)),
    }))
    .sort((left, right) => right.score - left.score);
}

function normalizeStatus(status) {
  return typeof status === "string" ? status.toUpperCase() : "";
}

function average(scores) {
  if (!Array.isArray(scores) || scores.length === 0) {
    return 0;
  }

  const total = scores.reduce((sum, score) => sum + score, 0);
  return total / scores.length;
}

function roundScore(score) {
  return Number(score.toFixed(4));
}

function sortByScoreDescending(left, right) {
  if (right.score === left.score) {
    return ALLOWED_EMOTIONS.indexOf(left.emotion) - ALLOWED_EMOTIONS.indexOf(right.emotion);
  }

  return right.score - left.score;
}

async function buildApiError(context, response) {
  const errorText = await response.text();
  return buildApiErrorFromText(context, response.status, response.statusText, errorText);
}

function buildApiErrorFromText(context, status, statusText, errorText) {
  const parsedMessage = parseErrorMessage(errorText);
  const statusSummary = `${status} ${statusText}`.trim();
  const details = parsedMessage ? ` ${parsedMessage}` : "";

  return new Error(`${context} (${statusSummary}).${details}`);
}

function shouldTreatAsEmptyAnalysis(jobDetails, utteranceEmotionScores) {
  if (utteranceEmotionScores.length > 0) {
    return false;
  }

  const numPredictions = Number(jobDetails?.state?.num_predictions ?? 0);
  return numPredictions === 0 || utteranceEmotionScores.length === 0;
}

function createEmptyAnalysisError(
  audioFile,
  predictionPayload,
  jobDetails,
  analysisMode,
  transcriptionLanguage
) {
  const extractedErrors = extractAnalysisErrors(predictionPayload);
  const numErrors = Number(jobDetails?.state?.num_errors ?? 0);
  const fileType = audioFile?.type?.toLowerCase() ?? "";
  const formatHint = isLessReliableSpeechFormat(fileType)
    ? " Try recording again or use a WAV, MP3, or MP4 file."
    : " Try a clearer recording or a WAV, MP3, or MP4 file.";
  const segmentationHint =
    analysisMode === "utterance"
      ? " Retrying with a different speech segmentation strategy."
      : "";
  const transcriptionFailure = extractedErrors.some((message) =>
    message.toLowerCase().includes("transcript confidence")
  );
  const languageHint = transcriptionLanguage
    ? " Try switching the speech language if the recording is in another language or mixed speech."
    : " Try setting the speech language before retrying if you're speaking Hindi, Hinglish, or another non-English language.";
  const serviceHint = transcriptionFailure
    ? "We couldn't confidently recognize spoken words in this audio."
    : numErrors > 0 || extractedErrors.length > 0
      ? "The audio was uploaded, but no usable speech emotion result could be produced."
      : "No usable speech emotion result could be produced from this audio.";
  saveAnalysisDebugSnapshot({
    analysisMode,
    audioFileName: audioFile?.name ?? "",
    audioFileType: audioFile?.type ?? "",
    extractedErrors,
    jobDetails,
    predictionPayload,
    transcriptionLanguage,
  });
  const error = new Error(
    `${serviceHint}${transcriptionFailure ? languageHint : formatHint}${segmentationHint}`
  );
  error.code = "EMPTY_ANALYSIS";

  return error;
}

function extractAnalysisErrors(predictionPayload) {
  const sourceItems = Array.isArray(predictionPayload)
    ? predictionPayload
    : predictionPayload
      ? [predictionPayload]
      : [];
  const collectedMessages = [];

  for (const sourceItem of sourceItems) {
    collectErrorMessages(sourceItem?.errors, collectedMessages);
    collectErrorMessages(sourceItem?.results?.errors, collectedMessages);
    collectErrorMessages(sourceItem?.response?.errors, collectedMessages);
  }

  return collectedMessages;
}

function collectErrorMessages(errorsValue, collectedMessages) {
  const errorItems = toArray(errorsValue);

  for (const errorItem of errorItems) {
    const message =
      errorItem?.message ??
      errorItem?.detail ??
      errorItem?.error ??
      errorItem?.reason;

    if (typeof message === "string" && message.trim()) {
      collectedMessages.push(message.trim());
    }
  }
}

function isLessReliableSpeechFormat(fileType) {
  return fileType.includes("webm") || fileType.includes("ogg") || fileType.includes("m4a");
}

function downmixToMono(audioBuffer) {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleCount = audioBuffer.length;
  const monoSamples = new Float32Array(sampleCount);

  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const channelData = audioBuffer.getChannelData(channelIndex);

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      monoSamples[sampleIndex] += channelData[sampleIndex] / channelCount;
    }
  }

  return monoSamples;
}

function trimSilence(samples, sampleRate) {
  let startIndex = 0;
  let endIndex = samples.length - 1;

  while (startIndex < samples.length && Math.abs(samples[startIndex]) < SILENCE_THRESHOLD) {
    startIndex += 1;
  }

  while (endIndex > startIndex && Math.abs(samples[endIndex]) < SILENCE_THRESHOLD) {
    endIndex -= 1;
  }

  if (startIndex >= endIndex) {
    return samples;
  }

  const paddingInSamples = Math.floor(sampleRate * SILENCE_PADDING_SECONDS);
  const safeStartIndex = Math.max(0, startIndex - paddingInSamples);
  const safeEndIndex = Math.min(samples.length, endIndex + paddingInSamples + 1);

  return samples.slice(safeStartIndex, safeEndIndex);
}

function normalizeSamples(samples) {
  let peak = 0;

  for (let index = 0; index < samples.length; index += 1) {
    peak = Math.max(peak, Math.abs(samples[index]));
  }

  if (peak === 0 || peak >= TARGET_PEAK_AMPLITUDE) {
    return samples;
  }

  const gain = TARGET_PEAK_AMPLITUDE / peak;
  const normalizedSamples = new Float32Array(samples.length);

  for (let index = 0; index < samples.length; index += 1) {
    normalizedSamples[index] = Math.max(-1, Math.min(1, samples[index] * gain));
  }

  return normalizedSamples;
}

function resampleLinear(samples, inputSampleRate, outputSampleRate) {
  if (inputSampleRate === outputSampleRate) {
    return samples;
  }

  const outputLength = Math.max(1, Math.round(samples.length * outputSampleRate / inputSampleRate));
  const resampled = new Float32Array(outputLength);
  const sampleRateRatio = inputSampleRate / outputSampleRate;

  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const sourceIndex = outputIndex * sampleRateRatio;
    const leftIndex = Math.floor(sourceIndex);
    const rightIndex = Math.min(leftIndex + 1, samples.length - 1);
    const interpolation = sourceIndex - leftIndex;
    const leftValue = samples[leftIndex] ?? 0;
    const rightValue = samples[rightIndex] ?? leftValue;

    resampled[outputIndex] = leftValue + (rightValue - leftValue) * interpolation;
  }

  return resampled;
}

function createWavBlob(samples, sampleRate) {
  const wavBuffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(wavBuffer);

  writeAsciiString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
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
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;

  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
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

function replaceFileExtension(filename, nextExtension) {
  if (typeof filename !== "string" || !filename.trim()) {
    return `audio-input.${nextExtension}`;
  }

  return filename.includes(".")
    ? filename.replace(/\.[^.]+$/, `.${nextExtension}`)
    : `${filename}.${nextExtension}`;
}

function saveAnalysisDebugSnapshot(debugPayload) {
  if (typeof window === "undefined") {
    return;
  }

  window.__emotionAnalysisDebug = debugPayload;

  if (import.meta.env.DEV) {
    console.debug("Emotion analysis debug:", debugPayload);
  }
}

function dedupePreservingOrder(values) {
  const seen = new Set();
  const deduped = [];

  for (const value of values) {
    const key = value ?? "__AUTO__";

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    deduped.push(value);
  }

  return deduped;
}

function parseErrorMessage(errorText) {
  if (!errorText) {
    return "";
  }

  try {
    const parsed = JSON.parse(errorText);
    return (
      parsed?.message ??
      parsed?.detail ??
      parsed?.error ??
      parsed?.errors?.[0]?.message ??
      errorText
    );
  } catch {
    return errorText;
  }
}

function toArray(value) {
  return Array.isArray(value) ? value : [];
}

function shouldRetryWithSingularFileField(status, errorText) {
  return status === 400 && errorText.includes("unknown field `file[]`");
}

function isRetryableEmptyAnalysis(error) {
  return Boolean(error && typeof error === "object" && error.code === "EMPTY_ANALYSIS");
}

function delay(durationMs) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}
