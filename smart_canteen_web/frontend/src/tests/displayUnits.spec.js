import { describe, expect, it } from 'vitest'
import {
  displayQuantity,
  displayUnit,
  formatDisplayQuantity,
} from '@/utils/displayUnits'

describe('销量展示单位', () => {
  it('克单位的原始50克计量数量换算为总克数且不改变传入原值', () => {
    const raw = 1.12
    expect(displayQuantity(raw, '克')).toBe(56)
    expect(displayUnit('克')).toBe('克')
    expect(formatDisplayQuantity(raw, '克')).toBe('56')
    expect(raw).toBe(1.12)
  })

  it('非克单位不转换', () => {
    expect(displayQuantity(125, '份')).toBe(125)
    expect(displayUnit('份')).toBe('份')
    expect(formatDisplayQuantity(125, '份')).toBe('125')
  })
})
