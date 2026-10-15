export const WEEKDAYS = [
  '星期日',
  '星期一',
  '星期二',
  '星期三',
  '星期四',
  '星期五',
  '星期六',
]

export function weekdayFromDate(dateText) {
  if (!dateText) return ''
  const [year, month, day] = dateText.split('-').map(Number)
  const date = new Date(Date.UTC(year, month - 1, day))
  return WEEKDAYS[date.getUTCDay()]
}

export function applyHolidayLinks(form) {
  if (!form.school_break) form.school_break_type = null
  if (!form.statutory_holiday) form.statutory_holiday_name = null
  return form
}

export function recommendationPayload(form, draft = {}) {
  const payload = {
    dataset_id: form.dataset_id,
    canteen: form.canteen,
    prediction_date: form.prediction_date,
    meal: form.meal,
    school_break: form.school_break,
    school_break_type: form.school_break ? form.school_break_type : null,
    statutory_holiday: form.statutory_holiday,
    statutory_holiday_name: form.statutory_holiday
      ? form.statutory_holiday_name
      : null,
    adjusted_workday: form.adjusted_workday,
    temperature_c: Number(form.temperature_c),
    humidity_pct: Number(form.humidity_pct),
    precipitation_mm: Number(form.precipitation_mm),
    wind_speed_kmh: Number(form.wind_speed_kmh),
    opportunity_refresh: Number(form.opportunity_refresh || 0),
    low_frequency_category: form.low_frequency_category,
    low_frequency_count: Number(form.low_frequency_count),
    previous_proposal_id: draft.previousProposalId || null,
    selected_low_frequency_dishes: draft.selectedLowFrequencyDishes || [],
  }
  return payload
}
