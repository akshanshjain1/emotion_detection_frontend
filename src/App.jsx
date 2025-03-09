import { useState,useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import EmotionDetection from './pages/emotion'
import Menu from './components/specific/menu'
import { LucideDoorOpen, MenuIcon, X } from 'lucide-react'
import CodeViewer from './components/specific/codeviewer'

function App() {
  
const [menu,setopenmenu]=useState(false)
const [viewcode,setviewcode]=useState(false);
const [code, setCode] = useState("");
const handleclosecode=()=>{
  setviewcode(false)
}
const handleopenmenu = () => {
  setopenmenu((prev)=>!prev);
};
useEffect(() => {
  fetch('/training_code.py')
  .then((response) => response.text())
  .then((data) => setCode(data)) // This should show the correct Python code
  .catch((error) => console.error("Error loading file:", error));

    
   
}, []);
  return (
    <div className="relative w-screen h-screen overflow-hidden">
      
      <div className="absolute top-[5%] left-[2%] z-50 text-white">
    {!menu && <MenuIcon onClick={handleopenmenu} />}
    {menu && <X onClick={handleopenmenu} />}
  </div>
  <div
    className={`fixed top-0 left-0 h-full w-64 bg-white shadow-lg transform ${
      menu ? "translate-x-0" : "-translate-x-full"
    } transition-transform duration-300 ease-in-out`}
  >
    <Menu viewcode={viewcode} setviewcode={setviewcode}/>
  </div>
  {viewcode && (
  <div className="fixed inset-0 flex justify-center items-center bg-black/50">
    <div className="w-[90%] max-w-3xl">
      <CodeViewer 
        code={code} 
        language="python"
        onClose={handleclosecode}
      />
    </div>
  </div>
)}

  {/* Main Content */}
  <div className="w-full h-full">
    <EmotionDetection menu={menu} setmenu={setopenmenu} />
  </div>
</div>

  )
}

export default App
