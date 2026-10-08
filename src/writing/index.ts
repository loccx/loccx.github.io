/**
 * Writing store: posts live as .md files in src/writing/posts/.
 * `[[slug]]` inside a post links to another post and draws an edge in the graph.
 */

export interface Post {
  slug: string;
  title: string;
  body: string;
  /** slugs this post links to */
  links: string[];
  /** slugs that link to this post */
  backlinks: string[];
}

const raw = import.meta.glob("./posts/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function slugOf(path: string): string {
  return path.split("/").pop()!.replace(/\.md$/, "");
}

function titleOf(slug: string, md: string): string {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : slug;
}

const posts = new Map<string, Post>();

for (const [path, md] of Object.entries(raw)) {
  const slug = slugOf(path);
  const links = [...md.matchAll(/\[\[([^\]]+)\]\]/g)].map((m) => m[1].trim());
  posts.set(slug, {
    slug,
    title: titleOf(slug, md),
    body: md,
    links,
    backlinks: [],
  });
}

for (const p of posts.values()) {
  for (const target of p.links) {
    posts.get(target)?.backlinks.push(p.slug);
  }
}

export function allPosts(): Post[] {
  return [...posts.values()];
}

export function getPost(slug: string): Post | undefined {
  return posts.get(slug);
}
