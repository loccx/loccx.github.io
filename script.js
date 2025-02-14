const text = "hello world";
let index = 0;

function typeWriter() {
    if (index < text.length) {
        document.getElementById("letter").innerHTML += text.charAt(index);
        index++;
        setTimeout(typeWriter, 150);
    }
}

function restartTyping() {
    index = 0;
    document.getElementById("letter").innerHTML = "";
    typeWriter();
}

window.onload = typeWriter;

