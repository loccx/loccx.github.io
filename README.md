# loccx.github.io

personal site. bun + typescript + vite + tailwind, deployed to github pages.

## dev

```sh
bun install
bun run dev      # local dev server
bun run build    # type-check + build to dist/
```

## structure

- `src/ui/manager.ts` — interface registry: `ui.register(id, factory)`, `ui.open(id)`
- `src/interfaces/` — one file per interface; register in `src/main.ts`
- `src/main.ts` — entry point

## deploy

push to `main` → github actions builds and deploys to https://loccx.github.io
