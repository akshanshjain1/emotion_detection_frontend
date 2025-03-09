export function Button({ children, className = "", ...props }) {
    return (
      <button
        className={`px-4 py-2 bg-blue-500 text-black-700 rounded-lg hover:bg-blue-700 transition ${className}`}
        {...props}
      >
        {children}
      </button>
    );
  }
  