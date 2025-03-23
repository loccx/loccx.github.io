import { useState, useEffect } from 'react';

function App() {
  const [para, setPara] = useState('');
  const paragraph = `                                    o$$
                                  $$$$$              ooo
                                $$$$$$$$          o$$$\" \"\"oo
                              o$$$\"$$\"$$ o\"$$$\"oo$\"$$$  $$o $
                            o$$$$ $$$ o $o\"     o\"\"\"\"\"\"$o\"$$oo\"$
                           o$$$o$o$$$\"  $o$$$$$\"\"\" oo$$$$$o$$$ \"
                           $$$\"$$$$$$ ooo$$\"\"$\"  o$$$$$o \"$o\"$$
                           $$$ o\"$$$$$\"  $$$\"ooo$ $$$\"\"$$o$$\" $o $
                          \"$$$ooo$$$$   $oo$$\"\"$$$$$$  $o$\"$ $\"$ $
                           \"$$\"$$$$$\"   $$$    o$$$$oo\"\"o$\"  $$ $ $
                            $$\"o$\"\"$    $$$$    \"$$$$$o$$\"  o$\"$$\"$
                             \"\"\" $$$    $$$$o    $$$o$o \"\"$$$  \"$
                               oo$\"$$    $$$$$\" \"$$$$ $$ooooo\"\"\"\"\"
                             o$$\"$\" $$o  $$$$$$   \"o$$o$
                           o$\"\" $$oo$$$$o$$$$$$$$$o$$$$
                         o\"$\" o$$\"  \"$$$$$$$\"$$\"$$$$$$$
                        o$$o$\"\"\"$o    \"$$$o\"o\"o$$\"    \\"\\"
                       $$$\"\"     \"\"o oooo\"\"\"\"\"\"
                    $o$$\"          $$$$$$$$o
                   o$$$            $\"$$$$o$\"\"$o
                  o$$$\"           $$$$$$$$oo\"\" $
                 o$oo$            $$\"$$$$$$$$\"$\"$
                o$o\"$               \"$$$$$$$$$$ oo$
                oo$$\"               \"\"$ $$$$$$$$$ $
                $$\"$$                  \"ooo$\"\"$$$$$o
                $$$\"                      \"\"$$o$\"$o\"
                $$$                           \"\"\"\"$
                $$$
                $$
                $$
    oo         $$$
   $$$         $$$  oooo$$ooo$o$oo
  o$$$$o       $$o$$$$$$$$$$$$$$\"
  $$$$$$o      $$$$$$$$$$$$$$$$\"
 $$$$$$$$o     o$$$$$$$$$o$$$$\"
 $$$$$$$$$o   $$$$$$$$$$$$$$\"
$$$$$$$$$$$$ o$$$$$$$$$$$$\"
$$$$$$$$$$$$ oo$\"$$$$$$\"\"\"
 $$$$$$$$$$$$$\"$\n
 \"$$$$$$$$$$$  $\n
   \"\"$$$$\"$$$$ \"\n
          $$ $$\"\n
          $$ $$\n
          $$ $$\n
          \"$$$$\n
           \"$$$\n`;

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      setPara((prev) => prev + paragraph[index]);
      index++;
      if (index === paragraph.length - 1) {
        clearInterval(interval);
      }
    }, 10);

    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <span data-typewriter='on'>happy 1 month bae 🧍🏻 here is a rose</span>
      <p style={{ whiteSpace: 'pre' }}>
        {para}
      </p>
    </>
  );
}

export default App;