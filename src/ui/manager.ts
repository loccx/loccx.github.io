/**
 * Interface registry — the seam between "the site" and "how it's presented".
 *
 * An Interface is anything that renders into a root element. Register them by
 * id, open/close them by id. Add a new file in src/interfaces/, register it in
 * main.ts, and it can be swapped in without touching anything else.
 */

export interface Interface {
  readonly id: string;
  mount(root: HTMLElement): void;
  unmount?(): void;
}

export type InterfaceFactory = () => Interface;

export class InterfaceManager {
  private factories = new Map<string, InterfaceFactory>();
  private active: { iface: Interface; root: HTMLElement } | null = null;
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

  open(id: string) {
    const factory = this.factories.get(id);
    if (!factory) throw new Error(`no interface registered as '${id}'`);
    this.close();
    const iface = factory();
    const root = document.createElement("div");
    root.dataset.interface = id;
    this.host.appendChild(root);
    iface.mount(root);
    this.active = { iface, root };
  }

  close() {
    this.active?.iface.unmount?.();
    this.active?.root.remove();
    this.active = null;
  }
}
