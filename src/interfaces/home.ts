import type { Interface } from "../ui/manager";

/** landing interface — hub linking to the other interfaces */
export class HomeInterface implements Interface {
  readonly id = "home";

  constructor(private open: (id: string) => void) {}

  mount(root: HTMLElement) {
    root.innerHTML = `
      <main class="min-h-screen grid place-items-center bg-zinc-950 text-zinc-200">
        <div class="text-center space-y-4">
          <p class="text-4xl font-bold tracking-tight">key</p>
          <button data-open="graph"
            class="block mx-auto text-sm text-zinc-400 hover:text-amber-400 border border-zinc-800 hover:border-amber-400/50 rounded px-4 py-1.5 transition-colors">
            writing →
          </button>
        </div>
      </main>`;
    root.querySelector("[data-open]")!.addEventListener("click", () =>
      this.open("graph"),
    );
  }
}
