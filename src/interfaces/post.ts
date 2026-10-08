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
    wrap.className = "fixed inset-0 bg-zinc-950 text-zinc-200 overflow-y-auto";

    const body = post
      ? renderMarkdown(post.body)
      : `<p class="text-zinc-500">post not found: ${this.slug}</p>`;
    const backlinks = post?.backlinks.length
      ? `<div class="mt-10 pt-4 border-t border-zinc-800 text-sm text-zinc-500">
           linked from:
           ${post.backlinks
             .map(
               (s) =>
                 `<a href="#" data-slug="${s}" class="text-amber-400 hover:underline ml-2">${s}</a>`,
             )
             .join("")}
         </div>`
      : "";

    wrap.innerHTML = `
      <article class="max-w-2xl mx-auto px-6 py-10">
        <button id="post-back" class="text-zinc-500 hover:text-zinc-200 text-sm mb-8">
          ← graph
        </button>
        ${body}
        ${backlinks}
      </article>`;
    root.appendChild(wrap);

    wrap.querySelector("#post-back")!.addEventListener("click", this.back);
    wrap.querySelectorAll("a[data-slug]").forEach((a) => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        this.openPost((a as HTMLElement).dataset.slug!);
      });
    });
  }
}
