import type { Interface } from "../ui/manager";

/** placeholder landing interface — replace with whatever the site becomes */
export class HomeInterface implements Interface {
  readonly id = "home";

  mount(root: HTMLElement) {
    root.innerHTML = `
      <main class="min-h-screen grid place-items-center bg-zinc-950 text-zinc-200">
        <div class="text-center space-y-3">
          <p class="text-4xl font-bold tracking-tight">key</p>
          <p class="text-zinc-500 text-sm">under construction</p>
        </div>
      </main>`;
  }
}
