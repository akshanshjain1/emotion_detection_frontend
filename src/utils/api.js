const HF_API_URL = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition";
const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";

/**
 * Queries the Hugging Face Inference API with an audio blob.
 */
export async function queryHuggingFace(audioBlob, apiKey) {
    const response = await fetch(HF_API_URL, {
        headers: {
            Authorization: `Bearer ${apiKey}`,
        },
        method: "POST",
        body: audioBlob,
    });

    if (!response.ok) {
        throw new Error(`Hugging Face API Error: ${response.statusText}`);
    }

    const result = await response.json();

    if (Array.isArray(result) && result.length > 0) {
        const sorted = result.sort((a, b) => b.score - a.score);
        return sorted[0].label;
    } else if (result.error) {
        throw new Error(result.error);
    }

    throw new Error("Unexpected response format from Hugging Face");
}

/**
 * Generates a comprehensive AI response using Groq with streaming.
 */
export async function generateGroqStream(emotion, apiKey, onChunk, type = 'comprehensive') {
    let systemPrompt = "";
    let userPrompt = "";

    if (type === 'comprehensive') {
        systemPrompt = `You are an exceptionally wise and deeply empathetic emotional wellness guide. You combine the wisdom of ancient philosophers with modern psychology. Your words are carefully chosen to resonate deeply with the human heart. You speak with warmth, authenticity, and profound understanding. Never mention you're an AI - you are a caring companion on the listener's emotional journey.`;
        userPrompt = `The voice analysis reveals the listener is experiencing: ${emotion}

Craft a deeply meaningful response (4-5 sentences) that:
• Opens with a poetic acknowledgment of their emotional state that makes them feel truly seen and understood
• Shares a profound insight about what this emotion might be teaching them about their inner world
• Offers a gentle, transformative suggestion that honors their experience while guiding them toward emotional harmony
• Closes with words that leave them feeling uplifted and hopeful

Write with elegance, depth, and genuine care. Let your words be a warm embrace for their soul.`;
    } else if (type === 'quick') {
        systemPrompt = "You are a warm, wise friend offering a brief moment of emotional connection.";
        userPrompt = `Someone is feeling ${emotion}. Offer them a brief, heartfelt message (1-2 sentences) that acknowledges their emotion with genuine warmth.`;
    }

    try {
        const response = await fetch(GROQ_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${apiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userPrompt }
                ],
                model: "llama-3.3-70b-versatile",
                temperature: 0.7,
                max_tokens: 1024,
                stream: true,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Groq API Error Response:", errorText);
            throw new Error(`Groq API Error: ${errorText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");

            for (const line of lines) {
                if (line.startsWith("data: ") && line !== "data: [DONE]") {
                    try {
                        const json = JSON.parse(line.substring(6));
                        const content = json.choices[0]?.delta?.content || "";
                        if (content) {
                            onChunk(content);
                        }
                    } catch (e) {
                        // Silently handle parsing errors
                    }
                }
            }
        }
    } catch (error) {
        console.error("Error in generateGroqStream:", error);
        throw error;
    }
}

/**
 * Get emotion-specific wellness tips using Groq
 */
export async function getWellnessTips(emotion, apiKey) {
    try {
        const response = await fetch(GROQ_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${apiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                messages: [
                    {
                        role: "system",
                        content: "You are an expert in emotional wellness and psychology. Provide practical, science-backed tips."
                    },
                    {
                        role: "user",
                        content: `Give 3 brief, actionable tips for someone feeling ${emotion}. Format as a numbered list. Each tip should be 1-2 sentences.`
                    }
                ],
                model: "llama-3.3-70b-versatile",
                temperature: 0.6,
                max_tokens: 500,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Groq API Error:", errorText);
            throw new Error("Failed to get wellness tips");
        }

        const data = await response.json();
        return data.choices[0].message.content;
    } catch (error) {
        console.error("Error in getWellnessTips:", error);
        throw error;
    }
}

/**
 * Get an uplifting affirmation based on the emotion
 */
export async function getAffirmation(emotion, apiKey) {
    try {
        const response = await fetch(GROQ_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${apiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                messages: [
                    {
                        role: "system",
                        content: "You create powerful, personalized affirmations that resonate deeply with people's emotional states."
                    },
                    {
                        role: "user",
                        content: `Create a powerful, personalized affirmation for someone feeling ${emotion}. Make it inspirational and empowering. Just the affirmation, no explanation.`
                    }
                ],
                model: "llama-3.3-70b-versatile",
                temperature: 0.8,
                max_tokens: 100,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Groq API Error:", errorText);
            throw new Error("Failed to get affirmation");
        }

        const data = await response.json();
        return data.choices[0].message.content;
    } catch (error) {
        console.error("Error in getAffirmation:", error);
        throw error;
    }
}
