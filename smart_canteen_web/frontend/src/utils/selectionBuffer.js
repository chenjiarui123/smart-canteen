import { displayQuantity, displayUnit } from './displayUnits'

const USER_CATEGORIES = new Set(['荤菜', '素菜', '主食', '甜品', '水果'])

export function dishKey(name) {
  return String(name || '').trim().toLocaleLowerCase('zh-CN')
}

export function makeSelectionRecord(item) {
  const source = item.recommendation_source || 'high_frequency'
  const originalQty = Number(
    source === 'low_frequency_opportunity'
      ? (item.suggested_trial_qty ?? item.pred_qty)
      : item.pred_qty,
  )
  return {
    dish_name: item.dish_name,
    recommendation_source: source,
    source_label: source === 'high_frequency' ? '常用菜品推荐' : '潜力菜',
    dish_category: USER_CATEGORIES.has(item.dish_category)
      ? item.dish_category
      : '其他',
    original_pred_qty: originalQty,
    display_qty: displayQuantity(originalQty, item.sale_unit),
    display_unit: displayUnit(item.sale_unit),
    pred_orders: Number(item.pred_orders || 0),
    pred_rev: Number(item.pred_rev || 0),
  }
}

export function addSelection(records, item) {
  const key = dishKey(item.dish_name)
  if (records.some((record) => dishKey(record.dish_name) === key)) return records
  return [...records, makeSelectionRecord(item)]
}

export function removeSelection(records, dishName) {
  const key = dishKey(dishName)
  return records.filter((record) => dishKey(record.dish_name) !== key)
}
