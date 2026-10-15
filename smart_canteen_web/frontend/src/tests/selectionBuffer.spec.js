import { describe, expect, it } from 'vitest'
import {
  addSelection,
  makeSelectionRecord,
  removeSelection,
} from '@/utils/selectionBuffer'

const gramDish = {
  dish_name: '称重菜', recommendation_source: 'low_frequency_opportunity',
  dish_category: '素菜', sale_unit: '克', pred_qty: 100,
  suggested_trial_qty: 100, pred_orders: 3, pred_rev: 20,
}

describe('已选菜品缓冲区', () => {
  it('保存来源、原始量和展示量并防止重复', () => {
    const record = makeSelectionRecord(gramDish)
    expect(record.original_pred_qty).toBe(100)
    expect(record.display_qty).toBe(5000)
    expect(record.display_unit).toBe('克')
    const once = addSelection([], gramDish)
    const twice = addSelection(once, gramDish)
    expect(twice).toHaveLength(1)
  })

  it('仅在明确删除时移除', () => {
    const records = addSelection([], gramDish)
    expect(removeSelection(records, '称重菜')).toEqual([])
  })
})
