<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { deploymentUrl } from '@/utils/deploymentUrl'
import {
  createRecommendation,
  errorMessage,
  finalizeRecommendation,
  getContext,
  getOptions,
} from '@/api/client'
import {
  applyDateContext,
  applyHolidayLinks,
  recommendationPayload,
  resetDateContext,
  weekdayFromDate,
} from '@/utils/formRules'
import {
  displayQuantity,
  displayUnit,
  formatDisplayQuantity,
} from '@/utils/displayUnits'
import {
  addSelection,
  dishKey,
  makeSelectionRecord,
  removeSelection,
} from '@/utils/selectionBuffer'

const formRef = ref()
const loadingOptions = ref(true)
const loadingContext = ref(false)
const generating = ref(false)
const finalizing = ref(false)
const contextNotice = ref('')
const result = ref(null)
const selectionBuffer = ref([])
const redistributeToDemand = ref(false)
const finalCategoryFilter = ref('全部')
const highCategoryFilter = ref('全部')
const highMinQty = ref(null)
const highMaxQty = ref(null)
let contextRequestSequence = 0

const options = reactive({
  datasets: [],
  meals: [],
  yes_no: [],
  school_break_types: [],
  statutory_holiday_names: [],
  dish_categories: ['全部', '荤菜', '素菜', '主食', '甜品', '水果'],
  weather: {},
})

const form = reactive({
  dataset_id: '',
  canteen: '',
  prediction_date: '',
  meal: '',
  school_break: false,
  school_break_type: null,
  statutory_holiday: false,
  statutory_holiday_name: null,
  adjusted_workday: false,
  temperature_c: null,
  humidity_pct: null,
  precipitation_mm: null,
  wind_speed_kmh: null,
  opportunity_refresh: 0,
  low_frequency_category: '全部',
  low_frequency_count: 12,
})

const acceptedContext = reactive({
  dataset_id: '',
  prediction_date: '',
  meal: '',
})

const rules = {
  dataset_id: [{ required: true, message: '请选择食堂', trigger: 'change' }],
  prediction_date: [{ required: true, message: '请选择推荐日期', trigger: 'change' }],
  meal: [{ required: true, message: '请选择餐次', trigger: 'change' }],
  low_frequency_category: [{ required: true, message: '请选择潜力菜类别', trigger: 'change' }],
  low_frequency_count: [{
    validator: (_rule, value, callback) => {
      if (!Number.isInteger(value) || value < 1 || value > 60) {
        callback(new Error('推荐数量必须是1至60之间的整数'))
      } else callback()
    },
    trigger: ['change', 'blur'],
  }],
  school_break_type: [{
    validator: (_rule, value, callback) => {
      if (form.school_break && !value) callback(new Error('请选择校历假期类型'))
      else callback()
    },
    trigger: 'change',
  }],
  statutory_holiday_name: [{
    validator: (_rule, value, callback) => {
      if (form.statutory_holiday && !value) callback(new Error('请选择法定节假日名称'))
      else callback()
    },
    trigger: 'change',
  }],
}

const weekday = computed(() => weekdayFromDate(form.prediction_date))
const weatherReady = computed(() =>
  ['temperature_c', 'humidity_pct', 'precipitation_mm', 'wind_speed_kmh']
    .every((key) => typeof form[key] === 'number' && Number.isFinite(form[key])),
)
const isDraft = computed(() => result.value?.status === 'draft')
const selectedCount = computed(() => selectionBuffer.value.length)
const selectedHighCount = computed(() => selectionBuffer.value.filter(
  (item) => item.recommendation_source === 'high_frequency',
).length)
const selectedLowCount = computed(() => selectedCount.value - selectedHighCount.value)
const selectedHighNames = computed(() => selectionBuffer.value
  .filter((item) => item.recommendation_source === 'high_frequency')
  .map((item) => item.dish_name))
const selectedLowNames = computed(() => selectionBuffer.value
  .filter((item) => item.recommendation_source === 'low_frequency_opportunity')
  .map((item) => item.dish_name))
const selectedCandidateQty = computed(() => selectionBuffer.value.reduce(
  (sum, item) => sum + item.original_pred_qty,
  0,
))
const menuReferenceExceeded = computed(() => Boolean(
  result.value && selectedCount.value > result.value.summary.menu_capacity,
))
const highRangeValid = computed(() => (
  highMinQty.value == null
  || highMaxQty.value == null
  || Number(highMinQty.value) <= Number(highMaxQty.value)
))

function normalizedCategory(value) {
  return options.dish_categories.includes(value) && value !== '全部'
    ? value
    : '其他'
}

function itemDisplayQty(item, field = 'pred_qty') {
  return displayQuantity(item?.[field] ?? 0, item?.sale_unit)
}

const filteredHighMenu = computed(() => {
  if (!result.value || !highRangeValid.value) return []
  return result.value.high_frequency_menu.filter((item) => {
    const categoryMatches = highCategoryFilter.value === '全部'
      || normalizedCategory(item.dish_category) === highCategoryFilter.value
    const qty = itemDisplayQty(item)
    const aboveMin = highMinQty.value == null || qty >= Number(highMinQty.value)
    const belowMax = highMaxQty.value == null || qty <= Number(highMaxQty.value)
    return categoryMatches && aboveMin && belowMax
  })
})

const filteredMenu = computed(() => {
  if (!result.value) return []
  if (finalCategoryFilter.value === '全部') return result.value.menu
  return result.value.menu.filter(
    (item) => normalizedCategory(item.dish_category) === finalCategoryFilter.value,
  )
})

function weatherOptions(key) {
  return (options.weather[key]?.values || []).map((value) => ({
    value,
    label: `${value} ${options.weather[key]?.unit || ''}`,
  }))
}

function isBuffered(itemOrName) {
  const name = typeof itemOrName === 'string' ? itemOrName : itemOrName.dish_name
  const key = dishKey(name)
  return selectionBuffer.value.some((item) => dishKey(item.dish_name) === key)
}

function toggleCandidate(item, checked) {
  selectionBuffer.value = checked
    ? addSelection(selectionBuffer.value, item)
    : removeSelection(selectionBuffer.value, item.dish_name)
}

function deleteBuffered(dishName) {
  selectionBuffer.value = removeSelection(selectionBuffer.value, dishName)
}

function normalizeLowFrequencyCount(value) {
  if (!Number.isInteger(value) || value < 1 || value > 60) {
    form.low_frequency_count = 12
    ElMessage.warning('推荐数量已恢复为默认值12')
  }
}

function enforceHighRange(changedSide) {
  if (highRangeValid.value) return
  if (changedSide === 'min') highMaxQty.value = highMinQty.value
  else highMinQty.value = highMaxQty.value
  ElMessage.warning('最低预计销量不能大于最高预计销量，已自动调整')
}

function clearRecommendationState() {
  result.value = null
  selectionBuffer.value = []
  redistributeToDemand.value = false
  form.opportunity_refresh = 0
  highCategoryFilter.value = '全部'
  highMinQty.value = null
  highMaxQty.value = null
  finalCategoryFilter.value = '全部'
}

function invalidateDraftForContextChange() {
  if (!result.value) return
  clearRecommendationState()
  contextNotice.value = '天气或校历已修改，请重新生成推荐。'
}

async function approveContextChange() {
  if (selectionBuffer.value.length) {
    try {
      await ElMessageBox.confirm(
        '更换食堂、日期或餐次会清空当前已选菜品，是否继续？',
        '确认更换推荐条件',
        { confirmButtonText: '继续更换', cancelButtonText: '保留当前选择', type: 'warning' },
      )
    } catch (_error) {
      return false
    }
  }
  clearRecommendationState()
  return true
}

async function selectDataset(datasetId) {
  const previous = acceptedContext.dataset_id
  if (!(await approveContextChange())) {
    form.dataset_id = previous
    const oldDataset = options.datasets.find((item) => item.id === previous)
    form.canteen = oldDataset?.canonical_canteen || ''
    return
  }
  acceptedContext.dataset_id = datasetId
  const selected = options.datasets.find((item) => item.id === datasetId)
  form.canteen = selected?.canonical_canteen || ''
}

async function changeDate(dateText) {
  const previous = acceptedContext.prediction_date
  if (!(await approveContextChange())) {
    form.prediction_date = previous
    return
  }
  acceptedContext.prediction_date = dateText || ''
  await fillDateContext()
}

async function changeMeal(meal) {
  const previous = acceptedContext.meal
  if (!(await approveContextChange())) {
    form.meal = previous
    return
  }
  acceptedContext.meal = meal || ''
}

async function fillDateContext() {
  const requestSequence = ++contextRequestSequence
  const requestedDate = form.prediction_date
  contextNotice.value = ''
  resetDateContext(form)
  if (!requestedDate) {
    loadingContext.value = false
    return
  }
  loadingContext.value = true
  try {
    const context = await getContext(requestedDate)
    if (
      requestSequence !== contextRequestSequence
      || form.prediction_date !== requestedDate
    ) return
    applyDateContext(form, context)
    if (!context.found_calendar || !context.found_weather) {
      contextNotice.value = [
        !context.found_calendar ? '该日期没有校历记录' : '',
        !context.found_weather ? '该日期没有天气记录' : '',
      ].filter(Boolean).join('；') + '，请从下拉选项手动选择。'
    } else {
      contextNotice.value = '已自动回填天气和校历信息，可在下拉选项内修改。'
    }
  } catch (error) {
    if (requestSequence !== contextRequestSequence) return
    contextNotice.value = `自动回填失败：${errorMessage(error)}`
  } finally {
    if (requestSequence === contextRequestSequence) loadingContext.value = false
  }
}

watch(() => form.school_break, () => applyHolidayLinks(form))
watch(() => form.statutory_holiday, () => applyHolidayLinks(form))

async function submit(refresh = false) {
  try {
    await formRef.value.validate()
    if (!weatherReady.value) {
      ElMessage.warning('请完整选择四项天气数值')
      return
    }
    if (
      !Number.isInteger(form.low_frequency_count)
      || form.low_frequency_count < 1
      || form.low_frequency_count > 60
    ) {
      ElMessage.warning('潜力菜推荐数量必须是1至60之间的整数')
      return
    }
    if (!refresh && selectionBuffer.value.length) {
      try {
        await ElMessageBox.confirm(
          '重新生成会清空当前已选菜品，是否继续？',
          '确认重新生成',
          { confirmButtonText: '重新生成', cancelButtonText: '保留当前选择', type: 'warning' },
        )
      } catch (_error) {
        return
      }
      clearRecommendationState()
    }
    if (refresh && !isDraft.value) return
    const previousNonce = form.opportunity_refresh
    const previousProposalId = refresh ? result.value.proposal_id : null
    const retainedLow = refresh ? [...selectedLowNames.value] : []
    if (refresh) form.opportunity_refresh = Math.min(previousNonce + 1, 20)
    generating.value = true
    try {
      const response = await createRecommendation(recommendationPayload(form, {
        previousProposalId,
        selectedLowFrequencyDishes: retainedLow,
      }))
      result.value = response
      if (!refresh) {
        selectionBuffer.value = response.high_frequency_menu.map(makeSelectionRecord)
      }
      const message = response.summary.low_frequency_message
      ElMessage.success(refresh ? `潜力菜已刷新。${message}` : '推荐候选已生成，请在已选菜品区确认菜单')
    } catch (error) {
      form.opportunity_refresh = previousNonce
      throw error
    }
  } catch (error) {
    if (error !== false) ElMessage.error(errorMessage(error))
  } finally {
    generating.value = false
  }
}

async function confirmSelection() {
  if (!result.value || !isDraft.value) return
  if (!selectedHighCount.value) {
    ElMessage.warning('最终菜单至少保留一道高频菜单菜品；潜力菜只作为额外试餐项')
    return
  }
  if (!selectedCount.value) {
    ElMessage.warning('最终菜单至少选择一道菜')
    return
  }
  try {
    finalizing.value = true
    result.value = await finalizeRecommendation({
      proposal_id: result.value.proposal_id,
      selected_high_frequency: selectedHighNames.value,
      selected_low_frequency: selectedLowNames.value,
      redistribute_to_total_demand: redistributeToDemand.value,
    })
    finalCategoryFilter.value = '全部'
    ElMessage.success('菜单已通过搭配检查并保存')
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    finalizing.value = false
  }
}

onMounted(async () => {
  try {
    Object.assign(options, await getOptions())
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loadingOptions.value = false
  }
})
</script>

<template>
  <div>
    <div class="page-title">
      <div>
        <h1>菜单推荐</h1>
        <p>填写用餐条件，选择候选菜品，确认后保存菜单。</p>
      </div>
    </div>

    <el-card class="section-card" v-loading="loadingOptions">
      <h2 class="section-heading"><el-icon><Calendar /></el-icon>用餐条件</h2>
      <el-form ref="formRef" :model="form" :rules="rules" label-position="top">
        <div class="form-grid">
          <el-form-item label="食堂" prop="dataset_id">
            <el-select v-model="form.dataset_id" placeholder="请选择食堂" @change="selectDataset">
              <el-option v-for="item in options.datasets" :key="item.id" :label="item.label" :value="item.id" />
            </el-select>
          </el-form-item>
          <el-form-item label="日期" prop="prediction_date">
            <el-date-picker v-model="form.prediction_date" type="date" value-format="YYYY-MM-DD" format="YYYY-MM-DD" placeholder="选择推荐日期" style="width: 100%" @change="changeDate" />
          </el-form-item>
          <el-form-item label="星期">
            <div class="readonly-field">{{ weekday || '选择日期后自动计算' }}</div>
          </el-form-item>
          <el-form-item label="餐次" prop="meal">
            <el-select v-model="form.meal" placeholder="请选择餐次" @change="changeMeal">
              <el-option v-for="item in options.meals" :key="item" :label="item" :value="item" />
            </el-select>
          </el-form-item>

          <el-form-item label="是否校历寒暑假">
            <el-select v-model="form.school_break" @change="invalidateDraftForContextChange">
              <el-option v-for="item in options.yes_no" :key="String(item.value)" :label="item.label" :value="item.value" />
            </el-select>
          </el-form-item>
          <el-form-item v-if="form.school_break" label="校历假期类型" prop="school_break_type">
            <el-select v-model="form.school_break_type" placeholder="请选择" @change="invalidateDraftForContextChange">
              <el-option v-for="item in options.school_break_types" :key="item" :label="item" :value="item" />
            </el-select>
          </el-form-item>
          <el-form-item label="是否法定节假日">
            <el-select v-model="form.statutory_holiday" @change="invalidateDraftForContextChange">
              <el-option v-for="item in options.yes_no" :key="String(item.value)" :label="item.label" :value="item.value" />
            </el-select>
          </el-form-item>
          <el-form-item v-if="form.statutory_holiday" label="法定节假日名称" prop="statutory_holiday_name">
            <el-select v-model="form.statutory_holiday_name" placeholder="请选择" @change="invalidateDraftForContextChange">
              <el-option v-for="item in options.statutory_holiday_names" :key="item" :label="item" :value="item" />
            </el-select>
          </el-form-item>
          <el-form-item label="是否调休工作日">
            <el-select v-model="form.adjusted_workday" @change="invalidateDraftForContextChange">
              <el-option v-for="item in options.yes_no" :key="String(item.value)" :label="item.label" :value="item.value" />
            </el-select>
          </el-form-item>

          <el-form-item label="气温（℃）">
            <el-select-v2 v-model="form.temperature_c" :options="weatherOptions('temperature_c')" filterable placeholder="选择气温" @change="invalidateDraftForContextChange" />
          </el-form-item>
          <el-form-item label="相对湿度（%）">
            <el-select-v2 v-model="form.humidity_pct" :options="weatherOptions('humidity_pct')" filterable placeholder="选择湿度" @change="invalidateDraftForContextChange" />
          </el-form-item>
          <el-form-item label="降水量（毫米）">
            <el-select-v2 v-model="form.precipitation_mm" :options="weatherOptions('precipitation_mm')" filterable placeholder="选择降水量" @change="invalidateDraftForContextChange" />
          </el-form-item>
          <el-form-item label="10米风速（公里/小时）">
            <el-select-v2 v-model="form.wind_speed_kmh" :options="weatherOptions('wind_speed_kmh')" filterable placeholder="选择风速" @change="invalidateDraftForContextChange" />
          </el-form-item>
        </div>
        <el-alert v-if="contextNotice" :title="contextNotice" :type="contextNotice.startsWith('已') ? 'success' : 'warning'" :closable="false" show-icon style="margin-bottom: 16px" />
        <div class="action-row">
          <el-button type="primary" size="large" :loading="generating || loadingContext" :disabled="loadingOptions" @click="submit(false)">
            <el-icon><MagicStick /></el-icon>生成推荐菜单
          </el-button>
        </div>
      </el-form>
    </el-card>

    <el-card v-if="result" class="section-card">
      <h2 class="section-heading"><el-icon><DataAnalysis /></el-icon>推荐概览</h2>
      <div class="summary-grid">
        <div class="summary-item"><label>食堂</label><strong>{{ result.summary.canteen }}</strong></div>
        <div class="summary-item"><label>日期 / 餐次</label><strong>{{ result.summary.prediction_date }}</strong><small>{{ result.summary.meal }}</small></div>
        <div class="summary-item"><label>{{ isDraft ? '当前已选' : '最终菜品' }}</label><strong>{{ isDraft ? selectedCount : result.summary.recommended_count }}</strong><small>道</small></div>
        <div class="summary-item"><label>餐次预测总量</label><strong>{{ result.summary.pred_total_qty.toFixed(2) }}</strong></div>
        <div class="summary-item"><label>总预测订单</label><strong>{{ result.summary.pred_total_orders.toFixed(2) }}</strong></div>
        <div class="summary-item"><label>总预测销售额</label><strong>{{ result.summary.pred_total_rev.toFixed(2) }}</strong><small>元</small></div>
      </div>
      <div class="impact-strip">
        <el-tag effect="plain">历史：{{ result.summary.history_start }} ～ {{ result.summary.history_end }}</el-tag>
        <el-tag type="info" effect="plain">{{ result.summary.weather_impact }}</el-tag>
        <el-tag type="warning" effect="plain">{{ result.summary.holiday_impact }}</el-tag>
        <el-tag type="success" effect="plain">综合因子 ×{{ result.summary.final_combined_factor.toFixed(3) }}</el-tag>
        <el-tag v-if="result.summary.forecast_gap_days" type="info" effect="plain">距历史末日 {{ result.summary.forecast_gap_days }} 天</el-tag>
        <el-tag :type="menuReferenceExceeded ? 'warning' : 'success'" effect="plain">参考菜品数 {{ result.summary.menu_capacity }} 道</el-tag>
        <el-tag effect="plain">{{ result.summary.selection_status }}</el-tag>
      </div>
      <el-alert v-if="result.summary.cold_start_warning" :title="result.summary.cold_start_warning" type="warning" :closable="false" show-icon />
      <el-alert v-if="result.summary.forecast_gap_warning" :title="result.summary.forecast_gap_warning" type="warning" :closable="false" show-icon style="margin-top: 12px" />
      <el-alert v-if="result.summary.adjacent_warning" :title="result.summary.adjacent_warning" type="warning" :closable="false" show-icon style="margin-top: 12px" />
      <div v-if="result.summary.adjacent_repeat_rate != null" class="impact-strip">
        <el-tag :type="result.summary.adjacent_repeat_rate <= 0.8 ? 'success' : 'warning'">自动高频菜单相邻日重复率 {{ (result.summary.adjacent_repeat_rate * 100).toFixed(1) }}%</el-tag>
        <el-tag>同餐次候选替换 {{ result.summary.replacement_count }} 道</el-tag>
        <el-tag>前日对照：{{ result.summary.adjacent_reference_source === 'confirmed' ? '已确认菜单' : '已保存高频结果' }}</el-tag>
      </div>
      <el-alert v-if="result.low_frequency_warning" :title="result.low_frequency_warning" type="warning" :closable="false" show-icon style="margin-top: 12px" />
    </el-card>

    <el-card v-if="result && isDraft" class="section-card">
      <div class="selection-header">
        <div>
          <h2 class="section-heading"><el-icon><List /></el-icon>选择候选菜品</h2>
          <p>常用菜品默认选中；潜力菜作为额外试餐项，不占用常用菜品名额。</p>
        </div>
        <div class="selection-metrics">
          <span>已选 {{ selectedCount }} 道 · 参考 {{ result.summary.menu_capacity }} 道</span>
          <span>原始预测/试餐量合计 {{ selectedCandidateQty.toFixed(2) }}</span>
        </div>
      </div>
      <el-alert v-if="menuReferenceExceeded" title="当前菜品数超过参考值；潜力菜仍可作为额外试餐项加入，不会挤占常用菜品名额。" type="warning" :closable="false" show-icon style="margin-bottom: 14px" />

      <section class="selection-buffer">
        <div class="buffer-header">
          <div>
            <strong>已选菜品</strong>
            <small>刷新或筛选不会移除这里的菜品</small>
          </div>
          <div class="buffer-counts">
            <el-tag>共 {{ selectedCount }} 道</el-tag>
            <el-tag type="success">常用菜品 {{ selectedHighCount }} 道</el-tag>
            <el-tag type="warning">潜力菜 {{ selectedLowCount }} 道</el-tag>
          </div>
        </div>
        <el-table :data="selectionBuffer" border stripe max-height="310" empty-text="尚未选择菜品">
          <el-table-column prop="dish_name" label="菜品" min-width="150" />
          <el-table-column prop="dish_category" label="类别" width="90" />
          <el-table-column label="展示销量" width="130">
            <template #default="{ row }">{{ row.display_qty.toFixed(2) }} {{ row.display_unit }}</template>
          </el-table-column>
          <el-table-column prop="source_label" label="来源" width="132" />
          <el-table-column prop="pred_orders" label="预测订单" width="110"><template #default="{ row }">{{ row.pred_orders.toFixed(2) }}</template></el-table-column>
          <el-table-column prop="pred_rev" label="预测销售额" width="120"><template #default="{ row }">{{ row.pred_rev.toFixed(2) }}</template></el-table-column>
          <el-table-column label="操作" width="82" fixed="right">
            <template #default="{ row }"><el-button type="danger" link @click="deleteBuffered(row.dish_name)">删除</el-button></template>
          </el-table-column>
        </el-table>
      </section>

      <div class="dual-menu-grid">
        <section class="menu-column high-column">
          <div class="column-title">
            <div><strong>常用菜品推荐</strong><small>筛选只改变显示，不影响已选菜品</small></div>
            <el-tag type="success">{{ filteredHighMenu.length }} / {{ result.high_frequency_menu.length }} 道</el-tag>
          </div>
          <div class="candidate-controls high-filter-controls">
            <div><label>菜品类别</label><el-select v-model="highCategoryFilter"><el-option v-for="item in options.dish_categories" :key="item" :label="item" :value="item" /></el-select></div>
            <div><label>最低预计销量</label><el-input-number v-model="highMinQty" :min="0" :precision="2" :controls="false" placeholder="可不填" @change="enforceHighRange('min')" /></div>
            <div><label>最高预计销量</label><el-input-number v-model="highMaxQty" :min="0" :precision="2" :controls="false" placeholder="可不填" @change="enforceHighRange('max')" /></div>
          </div>
          <el-alert v-if="!highRangeValid" title="最低预计销量不能大于最高预计销量" type="error" :closable="false" style="margin-bottom: 10px" />
          <el-table :data="filteredHighMenu" stripe border height="560">
            <el-table-column label="选" width="54" fixed><template #default="{ row }"><el-checkbox :model-value="isBuffered(row)" @change="(value) => toggleCandidate(row, value)" /></template></el-table-column>
            <el-table-column prop="dish_name" label="菜品" min-width="150" fixed sortable />
            <el-table-column label="预测销量" width="138" sortable :sort-method="(a, b) => itemDisplayQty(a) - itemDisplayQty(b)"><template #default="{ row }">{{ formatDisplayQuantity(row.pred_qty, row.sale_unit) }} {{ displayUnit(row.sale_unit) }}</template></el-table-column>
            <el-table-column prop="score" label="评分" width="108" sortable><template #default="{ row }">{{ row.score.toFixed(2) }}</template></el-table-column>
            <el-table-column label="类别" width="90"><template #default="{ row }">{{ normalizedCategory(row.dish_category) }}</template></el-table-column>
            <el-table-column prop="reason" label="推荐理由" min-width="360"><template #default="{ row }"><div class="reason-cell">{{ row.reason }}</div></template></el-table-column>
          </el-table>
        </section>

        <section class="menu-column low-column">
          <div class="column-title">
            <div><strong>低频高销量潜力菜</strong><small>按当前条件发现可试餐菜品</small></div>
            <el-tag type="warning">{{ result.low_frequency_opportunities.length }} 道</el-tag>
          </div>
          <div class="candidate-controls opportunity-controls">
            <div><label>菜品类别</label><el-select v-model="form.low_frequency_category"><el-option v-for="item in options.dish_categories" :key="item" :label="item" :value="item" /></el-select></div>
            <div><label>推荐数量</label><el-input-number v-model="form.low_frequency_count" :min="1" :max="60" :step="1" :precision="0" step-strictly @change="normalizeLowFrequencyCount" /></div>
            <el-button v-if="result.low_frequency_status === 'ok'" :loading="generating" @click="submit(true)">换一批</el-button>
          </div>
          <el-alert :title="result.summary.low_frequency_message" :type="result.summary.low_frequency_shortage ? 'warning' : 'success'" :closable="false" style="margin-bottom: 10px" />
          <el-table :data="result.low_frequency_opportunities" stripe border height="560" empty-text="本次没有合格潜力菜，常用菜品仍可正常确认">
            <el-table-column label="选" width="54" fixed><template #default="{ row }"><el-checkbox :model-value="isBuffered(row)" @change="(value) => toggleCandidate(row, value)" /></template></el-table-column>
            <el-table-column prop="dish_name" label="菜品" min-width="150" fixed sortable />
            <el-table-column label="类别" width="90"><template #default="{ row }">{{ normalizedCategory(row.dish_category) }}</template></el-table-column>
            <el-table-column label="建议试餐量" width="145" sortable :sort-method="(a, b) => itemDisplayQty(a, 'suggested_trial_qty') - itemDisplayQty(b, 'suggested_trial_qty')"><template #default="{ row }">{{ formatDisplayQuantity(row.suggested_trial_qty, row.sale_unit) }} {{ displayUnit(row.sale_unit) }}</template></el-table-column>
            <el-table-column prop="score" label="潜力评分" width="108" sortable><template #default="{ row }">{{ row.score.toFixed(2) }}</template></el-table-column>
            <el-table-column prop="ordinary_sample_count" label="普通日期样本" width="112" sortable />
            <el-table-column label="普通日期均量" width="150" sortable :sort-method="(a, b) => itemDisplayQty(a, 'ordinary_avg_qty') - itemDisplayQty(b, 'ordinary_avg_qty')"><template #default="{ row }">{{ formatDisplayQuantity(row.ordinary_avg_qty, row.sale_unit) }} {{ displayUnit(row.sale_unit) }}</template></el-table-column>
            <el-table-column label="节日使用" width="105"><template #default="{ row }"><el-tag :type="row.holiday_score_used ? 'warning' : 'info'" size="small">{{ row.holiday_score_used ? row.matched_holiday_name : '未使用' }}</el-tag></template></el-table-column>
            <el-table-column prop="selection_penalty" label="选中降权" width="106" sortable><template #default="{ row }">×{{ row.selection_penalty.toFixed(3) }}</template></el-table-column>
            <el-table-column prop="recommendation_reason" label="推荐理由" min-width="420"><template #default="{ row }"><div class="reason-cell">{{ row.recommendation_reason }}</div></template></el-table-column>
          </el-table>
        </section>
      </div>

      <div class="finalize-bar">
        <div class="finalize-copy">
          <strong>菜品数和备餐量仅作运营参考</strong>
          <span v-if="redistributeToDemand">确认后按当餐预测总需求 {{ result.summary.pred_total_qty.toFixed(2) }} 重新分配所有选中菜的备餐量。</span>
          <span v-else>保留常用菜品原预测量，并把潜力菜建议试餐量额外加入；总备餐量可能高于当餐预测需求。</span>
        </div>
        <div class="finalize-actions">
          <label class="reallocation-control"><span>按当餐总需求重新分配备餐量</span><el-switch v-model="redistributeToDemand" :width="68" inline-prompt active-text="开启" inactive-text="关闭" /></label>
          <el-button type="primary" size="large" :loading="finalizing" :disabled="selectedCount === 0" @click="confirmSelection">确认并保存菜单</el-button>
        </div>
      </div>
    </el-card>

    <el-card v-if="result && !isDraft" class="section-card">
      <h2 class="section-heading"><el-icon><List /></el-icon>已确认菜单</h2>
      <div v-if="result.combination_check" class="impact-strip">
        <el-tag type="success">硬约束违规 0</el-tag>
        <el-tag :type="result.combination_check.pred_total_conserved ? 'success' : 'warning'">{{ result.combination_check.redistributed_to_total_demand ? '已按总需求重分配' : '保留原量 + 试餐量' }}</el-tag>
        <el-tag>常用菜品 {{ result.combination_check.selected_high_frequency_count }} 道</el-tag>
        <el-tag type="warning">潜力菜 {{ result.combination_check.selected_low_frequency_count }} 道</el-tag>
        <el-tag>总备餐量 {{ result.combination_check.total_preparation_qty.toFixed(2) }}</el-tag>
      </div>
      <el-alert v-if="result.combination_check?.soft_warnings?.length" :title="`搭配提示：${result.combination_check.soft_warnings.join('；')}`" type="warning" :closable="false" show-icon />
      <el-alert v-if="result.combination_check?.reference_warnings?.length" :title="`数量/备餐提醒：${result.combination_check.reference_warnings.join('；')}`" type="warning" :closable="false" show-icon style="margin-top: 10px" />
      <div class="table-toolbar">
        <div class="filters"><el-select v-model="finalCategoryFilter" style="width: 190px"><el-option v-for="item in options.dish_categories" :key="item" :label="item" :value="item" /></el-select></div>
        <div class="download-group"><el-button tag="a" :href="deploymentUrl(result.csv_download_url)" type="primary" plain>下载菜单 CSV</el-button><el-button tag="a" :href="deploymentUrl(result.markdown_download_url)" type="success" plain>下载理由报告</el-button></div>
      </div>
      <el-table :data="filteredMenu" stripe border height="620">
        <el-table-column type="index" width="58" label="#" />
        <el-table-column label="来源" width="132" fixed><template #default="{ row }"><el-tag :type="row.recommendation_source === 'high_frequency' ? 'success' : 'warning'">{{ row.recommendation_source === 'high_frequency' ? '常用菜品推荐' : '潜力菜' }}</el-tag></template></el-table-column>
        <el-table-column prop="dish_name" label="菜品名称" min-width="160" fixed sortable />
        <el-table-column label="单位" width="92"><template #default="{ row }">{{ displayUnit(row.sale_unit) }}</template></el-table-column>
        <el-table-column label="预测销量" width="128" sortable :sort-method="(a, b) => itemDisplayQty(a) - itemDisplayQty(b)"><template #default="{ row }">{{ formatDisplayQuantity(row.pred_qty, row.sale_unit) }}</template></el-table-column>
        <el-table-column prop="pred_orders" label="预测订单" width="118" sortable><template #default="{ row }">{{ row.pred_orders.toFixed(2) }}</template></el-table-column>
        <el-table-column prop="pred_rev" label="预测销售额" width="130" sortable><template #default="{ row }">{{ row.pred_rev.toFixed(2) }}</template></el-table-column>
        <el-table-column prop="score" label="评分" width="120" sortable><template #default="{ row }">{{ row.score.toFixed(2) }}</template></el-table-column>
        <el-table-column label="类别" width="100"><template #default="{ row }">{{ normalizedCategory(row.dish_category) }}</template></el-table-column>
        <el-table-column prop="main_ingredient" label="主要食材" width="120" />
        <el-table-column prop="cooking_method" label="制作方法" width="110" />
        <el-table-column prop="flavor" label="口味" width="100" />
        <el-table-column prop="reason" label="推荐理由" min-width="440"><template #default="{ row }"><div class="reason-cell">{{ row.reason }}</div></template></el-table-column>
      </el-table>
    </el-card>

    <el-card v-if="!result" class="section-card">
      <div class="empty-panel"><el-icon size="42"><Dish /></el-icon><p>填写用餐条件后生成推荐，再从候选菜品中确认菜单。</p></div>
    </el-card>
  </div>
</template>
