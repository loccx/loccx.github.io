/**
 * Interface registry + navigation stack — the seam between "the site" and
 * "how it's presented".
 *
 * An Interface is anything that renders into a root element. Register them by
 * id, `open(id, arg)` pushes one on top of the stack, `close()` (or Esc) pops
 * back to whatever was underneath. Add a new file in src/interfaces/,
 * register it in main.ts, and it can be swapped in without touching the rest.
 */

export interface Interface {
  readonly id: string;
  mount(root: HTMLElement): void;
  unmount?(): void;
}

export type InterfaceFactory = (arg?: unknown) => Interface;

export class InterfaceManager {
  private factories = new Map<string, InterfaceFactory>();
  private stack: { iface: Interface; root: HTMLElement }[] = [];
  private host: HTMLElement;

  constructor(parent: HTMLElement) {
    this.host = document.createElement("div");
    this.host.id = "ui-root";
    parent.appendChild(this.host);

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") this.close();
    });
  }

  register(id: string, factory: InterfaceFactory) {
    this.factories.set(id, factory);
  }

  open(id: string, arg?: unknown) {
    const factory = this.factories.get(id);
    if (!factory) throw new Error(`no interface registered as '${id}'`);
    const top = this.stack[this.stack.length - 1];
    if (top) top.root.style.display = "none";
    const iface = factory(arg);
    const root = document.createElement("div");
    root.dataset.interface = id;
    this.host.appendChild(root);
    iface.mount(root);
    this.stack.push({ iface, root });
  }

  /** pop the current interface, revealing the one underneath */
  close() {
    const top = this.stack.pop();
    if (!top) return;
    top.iface.unmount?.();
    top.root.remove();
    const prev = this.stack[this.stack.length - 1];
    if (prev) prev.root.style.display = "";
  }
}
