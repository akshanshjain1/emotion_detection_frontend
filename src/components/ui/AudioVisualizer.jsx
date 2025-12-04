import { useEffect, useRef } from "react";

export default function AudioVisualizer({ stream, isRecording }) {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const analyserRef = useRef(null);
    const sourceRef = useRef(null);

    useEffect(() => {
        if (!stream || !isRecording) {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            return;
        }

        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;

        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        analyserRef.current = analyser;
        sourceRef.current = source;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            if (!isRecording) return;

            animationRef.current = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;

                const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
                gradient.addColorStop(0, "#a855f7"); // Purple-500
                gradient.addColorStop(1, "#3b82f6"); // Blue-500

                ctx.fillStyle = gradient;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

                x += barWidth + 1;
            }
        };

        draw();

        return () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            if (sourceRef.current) sourceRef.current.disconnect();
            if (audioContext.state !== "closed") audioContext.close();
        };
    }, [stream, isRecording]);

    return (
        <canvas
            ref={canvasRef}
            width={300}
            height={100}
            className="w-full h-32 rounded-xl bg-black/20 backdrop-blur-sm shadow-inner"
        />
    );
}
