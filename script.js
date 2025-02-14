const text = "hello world"const { useState, useEffect } = React;

function Typewriter() {
  const [text, setText] = useState('');
  const fullText = 'hello world';
  let index = 0;

  useEffect(() => {
    const interval = setInterval(() => {
      setText((prev) => prev + fullText[index]);
      index++;
      if (index === fullText.length) clearInterval(interval);
    }, 150);

    return () => clearInterval(interval);
  }, []);

  return <div className="letter">{text}</div>;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Typewriter />);
;
