import { Undo } from "lucide-react";
import { useEffect, useState } from "react";
const emojis = ["😢", "😨", "😠", "😲", "😐", "🤢", "😊"];
const emotionLabels = [
  "Sad",
  "Fear",
  "Angry",
  "Pleasant_surprised",
  "Neutral",
  "Disgust",
  "Happy",
];

const EmojiSwitcher = ({ emotion = "", loading, setisloading,setemotion }) => {
  const [index, setIndex] = useState(0);
  
  useEffect(() => {
    if (emotion === "") {
      const interval = setInterval(() => {
        setIndex((prevIndex) => (prevIndex + 1) % emojis.length);
      }, 200);

      return () => clearInterval(interval);
    }
  }, [emotion]);
  const handleRetry = () => {
    setisloading(false);
    setemotion("")

  };

  return (
    <>
      {emotion === "" ? (
        <div className="flex flex-col justify-center items-center min-h-[60%] text-6xl font-bold">
          <div>{emojis[index]}</div>
          <div className="text-[1.1rem] font-semibold text-gray-600 mt-2">
            Detecting Emotion: {emotionLabels[index]}
          </div>
        </div>
      ) : (
        <div className="flex flex-col justify-center items-center min-h-[60%] text-6xl font-bold">
          <div>
            {
              emojis[
                emotionLabels.indexOf(
                  emotion.charAt(0) + emotion.slice(1).toLowerCase()
                )
              ]
            }
          </div>
          <div className="text-[1.1rem] font-semibold text-gray-600 mt-2 mb-[2%] w-fit">
            detected emotion:{" "}
            {
              emotionLabels[
                emotionLabels.indexOf(
                  emotion.charAt(0) + emotion.slice(1).toLowerCase()
                )
              ]
            }
          </div>
          <div>
            <button
              onClick={handleRetry}
              className="bg-yellow-500 hover:bg-yellow-700 py-2 rounded-xl flex gap-2 justify-center items-center px-2"
            >
              <Undo className="w-fit h-fit " /> <div className="text-xl">try again</div>
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default EmojiSwitcher;
