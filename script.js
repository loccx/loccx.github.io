const text = "hello world";
let index = 0;

function typeWriter() {
    if (index < text.length) {
        document.getElementById("letter").innerHTML += text.charAt(index);
        index++;
        setTimeout(typeWriter, 150);
    }
}

window.onload = typeWriter;

