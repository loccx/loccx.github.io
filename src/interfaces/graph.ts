import type { Interface } from "../ui/manager";
import { allPosts } from "../writing";

interface Node {
  slug: string;
  title: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

/**
 * Obsidian-style graph of the posts in src/writing/posts.
 * Nodes = posts, edges = [[wikilinks]]. Drag to move, click to read.
 */
export class GraphInterface implements Interface {
  readonly id = "graph";
  private raf = 0;
  private cleanup: (() => void)[] = [];

  constructor(private openPost: (slug: string) => void) {}

  mount(root: HTMLElement) {
    const posts = allPosts();
    const wrap = document.createElement("div");
    wrap.className = "fixed inset-0";
    wrap.innerHTML = `
      <div class="absolute top-4 left-5 muted select-none" style="font-size: 14px">
        writing · drag nodes · click to read · esc to go back
      </div>`;
    const canvas = document.createElement("canvas");
    canvas.className = "block w-full h-full touch-none";
    wrap.appendChild(canvas);
    root.appendChild(wrap);

    const ctx = canvas.getContext("2d")!;
    const dpr = Math.min(devicePixelRatio || 1, 2);

    const nodes: Node[] = posts.map((p) => ({
      slug: p.slug,
      title: p.title,
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      r: 10 + Math.min(p.links.length + p.backlinks.length, 6) * 2,
    }));
    const bySlug = new Map(nodes.map((n) => [n.slug, n]));
    const edges: [Node, Node][] = [];
    for (const p of posts) {
      for (const t of p.links) {
        const b = bySlug.get(t);
        if (b) edges.push([bySlug.get(p.slug)!, b]);
      }
    }

    const size = () => {
      canvas.width = wrap.clientWidth * dpr;
      canvas.height = wrap.clientHeight * dpr;
    };
    size();
    window.addEventListener("resize", size);
    this.cleanup.push(() => window.removeEventListener("resize", size));

    // seed positions on a circle
    nodes.forEach((n, i) => {
      const a = (i / Math.max(nodes.length, 1)) * Math.PI * 2;
      n.x = Math.cos(a) * 160 + (Math.random() - 0.5) * 40;
      n.y = Math.sin(a) * 160 + (Math.random() - 0.5) * 40;
    });

    let hovered: Node | null = null;
    let dragged: Node | null = null;
    let dragMoved = false;
    const toWorld = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      return {
        x: e.clientX - r.left - r.width / 2,
        y: e.clientY - r.top - r.height / 2,
      };
    };
    const pick = (wx: number, wy: number) =>
      nodes.find((n) => Math.hypot(n.x - wx, n.y - wy) < n.r + 6) ?? null;

    canvas.addEventListener("pointerdown", (e) => {
      const p = toWorld(e);
      dragged = pick(p.x, p.y);
      dragMoved = false;
      if (dragged) canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      const p = toWorld(e);
      if (dragged) {
        dragged.x = p.x;
        dragged.y = p.y;
        dragged.vx = dragged.vy = 0;
        dragMoved = true;
      } else {
        hovered = pick(p.x, p.y);
        canvas.style.cursor = hovered ? "pointer" : "default";
      }
    });
    canvas.addEventListener("pointerup", (e) => {
      if (dragged && !dragMoved) {
        const p = toWorld(e);
        const n = pick(p.x, p.y);
        if (n) this.openPost(n.slug);
      }
      dragged = null;
    });

    const step = () => {
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) {
            dx = Math.random() - 0.5;
            dy = Math.random() - 0.5;
            d2 = 1;
          }
          const f = Math.min(4000 / d2, 4);
          const d = Math.sqrt(d2);
          a.vx += (dx / d) * f;
          a.vy += (dy / d) * f;
          b.vx -= (dx / d) * f;
          b.vy -= (dy / d) * f;
        }
      }
      for (const [a, b] of edges) {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1;
        const f = (d - 140) * 0.02;
        a.vx += (dx / d) * f;
        a.vy += (dy / d) * f;
        b.vx -= (dx / d) * f;
        b.vy -= (dy / d) * f;
      }
      for (const n of nodes) {
        if (n !== dragged) {
          n.vx += -n.x * 0.005;
          n.vy += -n.y * 0.005;
          n.x += n.vx;
          n.y += n.vy;
          n.vx *= 0.85;
          n.vy *= 0.85;
        }
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.translate(w / 2, h / 2);

      ctx.strokeStyle = "#3f3f46";
      ctx.lineWidth = 1;
      for (const [a, b] of edges) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      ctx.font = '14px "BigBlue Terminal", monospace';
      ctx.textAlign = "center";
      for (const n of nodes) {
        const hot = n === hovered || n === dragged;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = hot ? "#5f87ff" : "#a1a1aa";
        ctx.fill();
        ctx.fillStyle = hot ? "#5f87ff" : "#71717a";
        ctx.fillText(n.title, n.x, n.y + n.r + 16);
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  unmount() {
    cancelAnimationFrame(this.raf);
    this.cleanup.forEach((f) => f());
    this.cleanup = [];
  }
}
