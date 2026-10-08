import "./style.css";
import { InterfaceManager } from "./ui/manager";
import { HomeInterface } from "./interfaces/home";

const app = document.querySelector<HTMLDivElement>("#app")!;
const ui = new InterfaceManager(app);

// register interfaces here — one line each, swap freely
ui.register("home", () => new HomeInterface());

ui.open("home");
