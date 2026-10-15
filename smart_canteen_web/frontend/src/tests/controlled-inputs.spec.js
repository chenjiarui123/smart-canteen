import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const recommendationPath = resolve(
  process.cwd(),
  'src/views/RecommendationView.vue',
)
const source = readFileSync(recommendationPath, 'utf8')

describe('业务输入受控', () => {
  it('推荐页使用日历和下拉框且不含可编辑文本框', () => {
    expect(source).toContain('<el-date-picker')
    expect(source).toContain('<el-select')
    expect(source).toContain('<el-select-v2')
    expect(source).not.toContain('<el-input ')
    expect(source).toContain('<el-input-number')
  })

  it('下拉选项来自后端 options 接口', () => {
    expect(source).toContain('await getOptions()')
    expect(source).toContain('options.datasets')
    expect(source).toContain('options.weather')
  })

  it('常用菜品与低频潜力菜分栏选择后由后端确认合并', () => {
    expect(source).toContain('result.high_frequency_menu')
    expect(source).toContain('result.low_frequency_opportunities')
    expect(source).toContain('finalizeRecommendation')
    expect(source).toContain('selected_high_frequency')
    expect(source).toContain('selected_low_frequency')
    expect(source).toContain('redistribute_to_total_demand')
    expect(source).toContain('按当餐总需求重新分配备餐量')
    expect(source).toContain('不占用常用菜品名额')
    expect(source).toContain('换一批')
    expect(source).toContain('selectionBuffer')
    expect(source).toContain('previousProposalId')
    expect(source).toContain('selectedLowFrequencyDishes')
    expect(source).toContain('low_frequency_category')
    expect(source).toContain('low_frequency_count')
    expect(source).toContain(':min="1" :max="60"')
    expect(source).toContain('最低预计销量')
    expect(source).toContain('最高预计销量')
    expect(source).toContain('普通日期均量')
    expect(source).toContain('普通日期样本')
  })

  it('不展示开发过程文案', () => {
    for (const text of [
      'Trial',
      '冻结高频',
      '本地确定性推理',
      '本地运行',
      '数据不离开电脑',
      '预测与评分未修改',
      '独立模块',
    ]) {
      expect(source).not.toContain(text)
    }
  })
})
