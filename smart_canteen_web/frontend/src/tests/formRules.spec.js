import { describe, expect, it } from 'vitest'
import {
  applyHolidayLinks,
  recommendationPayload,
  weekdayFromDate,
} from '@/utils/formRules'

describe('受控推荐表单', () => {
  it('使用固定中文星期映射且不依赖系统语言', () => {
    expect(weekdayFromDate('2026-01-01')).toBe('星期四')
    expect(weekdayFromDate('2026-07-29')).toBe('星期三')
  })

  it('关闭假期标记时清理联动字段', () => {
    const form = {
      school_break: false,
      school_break_type: '寒假',
      statutory_holiday: false,
      statutory_holiday_name: '元旦',
    }
    applyHolidayLinks(form)
    expect(form.school_break_type).toBeNull()
    expect(form.statutory_holiday_name).toBeNull()
  })

  it('天气字段在请求体中保持 number 类型', () => {
    const payload = recommendationPayload({
      dataset_id: 'a_hupan',
      canteen: 'A湖畔智慧餐厅',
      prediction_date: '2026-07-29',
      meal: '午餐',
      school_break: false,
      school_break_type: null,
      statutory_holiday: false,
      statutory_holiday_name: null,
      adjusted_workday: false,
      temperature_c: 25.1,
      humidity_pct: 70,
      precipitation_mm: 0,
      wind_speed_kmh: 5.2,
      opportunity_refresh: 3,
      low_frequency_category: '荤菜',
      low_frequency_count: 10,
    })
    expect(typeof payload.temperature_c).toBe('number')
    expect(typeof payload.humidity_pct).toBe('number')
    expect(typeof payload.precipitation_mm).toBe('number')
    expect(typeof payload.wind_speed_kmh).toBe('number')
    expect(payload.opportunity_refresh).toBe(3)
    expect(payload.low_frequency_category).toBe('荤菜')
    expect(payload.low_frequency_count).toBe(10)
    expect(payload.previous_proposal_id).toBeNull()
    expect(payload.selected_low_frequency_dishes).toEqual([])
  })

  it('刷新时携带后端草稿编号和全部已选潜力菜', () => {
    const payload = recommendationPayload({
      dataset_id: 'a_hupan', canteen: 'A湖畔智慧餐厅',
      prediction_date: '2026-07-29', meal: '午餐',
      school_break: false, statutory_holiday: false, adjusted_workday: false,
      temperature_c: 25, humidity_pct: 70, precipitation_mm: 0,
      wind_speed_kmh: 5, opportunity_refresh: 2,
      low_frequency_category: '水果', low_frequency_count: 4,
    }, {
      previousProposalId: 'a'.repeat(32),
      selectedLowFrequencyDishes: ['西瓜'],
    })
    expect(payload.previous_proposal_id).toBe('a'.repeat(32))
    expect(payload.selected_low_frequency_dishes).toEqual(['西瓜'])
  })
})
