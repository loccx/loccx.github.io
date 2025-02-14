import { useState, useEffect } from 'react';

function App() {
  const [para, setPara] = useState('');
  const paragraph = 'bitches';

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      setPara((prev) => prev + paragraph[index]);
      index++;
      if (index === paragraph.length - 1) {
        clearInterval(interval);
      }
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <span data-typewriter = "on">Hello World!</span>
      <p>
        {para}
      </p>
    </>
  );
}

export default App;
