import "./style.css";
import { InterfaceManager } from "./ui/manager";
import { HomeInterface } from "./interfaces/home";
import { GraphInterface } from "./interfaces/graph";
import { PostInterface } from "./interfaces/post";

const app = document.querySelector<HTMLDivElement>("#app")!;
const ui = new InterfaceManager(app);

// register interfaces here — one line each, swap freely
ui.register("home", () => new HomeInterface((id) => ui.open(id)));
ui.register("graph", () => new GraphInterface((slug) => ui.open("post", slug)));
ui.register(
  "post",
  (arg) =>
    new PostInterface(
      arg as string,
      (slug) => ui.open("post", slug),
      () => ui.close(),
    ),
);

ui.open("home");
