import type { Interface } from "../ui/manager";
import { getPost } from "../writing";
import { renderMarkdown } from "../writing/md";

/** reader for a single post; `arg` is the post slug */
export class PostInterface implements Interface {
  readonly id = "post";

  constructor(
    private slug: string,
    private openPost: (slug: string) => void,
    private back: () => void,
  ) {}

  mount(root: HTMLElement) {
    const post = getPost(this.slug);
    const wrap = document.createElement("div");
    wrap.className = "fixed inset-0 overflow-y-auto";

    const body = post
      ? renderMarkdown(post.body)
      : `<p class="muted">post not found: ${this.slug}</p>`;
    const backlinks = post?.backlinks.length
      ? `<hr /><p class="muted">linked from: ${post.backlinks
          .map((s) => `<a href="#" data-slug="${s}">${s}</a>`)
          .join(" ")}</p>`
      : "";

    wrap.innerHTML = `
      <article style="max-width: 72ch; padding: 1.5em 2ch">
        <p><a href="#" id="post-back">&lt; graph</a></p>
        ${body}
        ${backlinks}
      </article>`;
    root.appendChild(wrap);

    wrap.querySelector("#post-back")!.addEventListener("click", (e) => {
      e.preventDefault();
      this.back();
    });
    wrap.querySelectorAll("a[data-slug]").forEach((a) => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        this.openPost((a as HTMLElement).dataset.slug!);
      });
    });
  }
}
