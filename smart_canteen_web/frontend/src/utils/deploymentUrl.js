// Backend download URLs are relative to the backend root.
export function deploymentUrl(path, base = import.meta.env.BASE_URL) {
  if (!path) return undefined
  return base + path.replace(/^\//, '')
}
