# loccx.github.io

personal site. bun + vite, static page, deployed to github pages.

## dev

```sh
bun install
bun run dev      # local dev server
bun run build    # type-check + build to dist/
```

## structure

- `index.html` — the whole page
- `src/style.css` — styling
- `public/resume.pdf` — served at `/resume.pdf`

## deploy

push to `main` → github actions builds and deploys to https://loccx.github.io
