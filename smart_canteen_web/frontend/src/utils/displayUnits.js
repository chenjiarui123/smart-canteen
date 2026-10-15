export function isGramUnit(unit) {
  return String(unit ?? '').trim() === '克'
}

export function displayQuantity(quantity, unit) {
  const numeric = Number(quantity ?? 0)
  if (!isGramUnit(unit)) return numeric
  const grams = numeric * 50
  return Math.round((grams + Number.EPSILON) * 100) / 100
}

export function displayUnit(unit) {
  return isGramUnit(unit) ? '克' : String(unit || '份')
}

export function formatDisplayQuantity(quantity, unit) {
  const value = displayQuantity(quantity, unit)
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}
