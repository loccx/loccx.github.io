import type { Interface } from "../ui/manager";

/** landing interface — flat archival index page */
export class HomeInterface implements Interface {
  readonly id = "home";

  constructor(private open: (id: string) => void) {}

  mount(root: HTMLElement) {
    root.innerHTML = `
      <main style="max-width: 72ch; padding: 1.5em 2ch">
        <h1>key</h1>
        <p class="muted">personal site. everything here is plain text.</p>

        <h2>index</h2>
        <ul>
          <li><a href="#" data-open>writing/</a> — notes, obsidian-style graph</li>
          <li><a href="https://github.com/loccx">github.com/loccx</a></li>
        </ul>

        <h2>sample heading</h2>
        <p>
          this is sample body text to show the styling. paragraphs are
          left-aligned and monospaced, links look like
          <a href="https://github.com/loccx">this</a>, inline code looks like
          <code>code</code>, and lists look like:
        </p>
        <ul>
          <li>item one</li>
          <li>item two</li>
        </ul>
        <pre>a code block
spans lines
  keeps indent</pre>

        <hr />
        <p class="muted">last updated: recently</p>
      </main>`;
    root.querySelector("[data-open]")!.addEventListener("click", (e) => {
      e.preventDefault();
      this.open("graph");
    });
  }
}
