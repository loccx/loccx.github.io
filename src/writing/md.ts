/**
 * Minimal markdown → html for posts. Handles headings, bold/italic, inline and
 * fenced code, links, lists, and [[wikilinks]] (emitted as <a data-slug>).
 * Everything is HTML-escaped first; this is not a full renderer on purpose.
 */

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inline(s: string): string {
  return s
    .replace(/`([^`]+)`/g, '<code class="px-1 rounded bg-zinc-800">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(
      /\[\[([^\]]+)\]\]/g,
      '<a href="#" data-slug="$1" class="text-amber-400 hover:underline">$1</a>',
    )
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener" class="text-amber-400 hover:underline">$1</a>',
    );
}

export function renderMarkdown(md: string): string {
  const blocks: string[] = [];
  let para: string[] = [];
  let list: string[] = [];
  let code = false;
  let codeBuf: string[] = [];

  const flushPara = () => {
    if (para.length) {
      blocks.push(`<p class="my-3 leading-relaxed">${inline(para.join(" "))}</p>`);
      para = [];
    }
  };
  const flushList = () => {
    if (list.length) {
      blocks.push(
        `<ul class="my-3 list-disc pl-6 space-y-1">${list
          .map((i) => `<li>${inline(i)}</li>`)
          .join("")}</ul>`,
      );
      list = [];
    }
  };

  for (const raw of escapeHtml(md).split("\n")) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) {
      if (code) {
        blocks.push(
          `<pre class="my-3 p-3 rounded bg-zinc-800 overflow-x-auto text-sm"><code>${codeBuf.join("\n")}</code></pre>`,
        );
        codeBuf = [];
        code = false;
      } else {
        flushPara();
        flushList();
        code = true;
      }
      continue;
    }
    if (code) {
      codeBuf.push(line);
      continue;
    }
    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flushPara();
      flushList();
      const level = h[1].length;
      const cls = [
        "text-3xl font-bold mt-6 mb-3",
        "text-2xl font-semibold mt-5 mb-2",
        "text-xl font-semibold mt-4 mb-2",
        "text-lg font-semibold mt-3 mb-1",
      ][level - 1];
      blocks.push(`<h${level} class="${cls}">${inline(h[2])}</h${level}>`);
      continue;
    }
    if (line.startsWith("- ") || line.startsWith("* ")) {
      flushPara();
      list.push(line.slice(2));
      continue;
    }
    para.push(line);
  }
  flushPara();
  flushList();
  return blocks.join("\n");
}
