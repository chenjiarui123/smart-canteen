<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  commitPreview,
  errorMessage,
  getOptions,
  previewUpload,
} from '@/api/client'

const options = reactive({ datasets: [] })
const kind = ref('sales')
const datasetId = ref('')
const uploadRef = ref(null)
const uploadFiles = ref([])
const selectedFile = ref(null)
const sheetName = ref('')
const preview = ref(null)
const previewing = ref(false)
const committing = ref(false)
const progress = ref(0)

const uploadTitle = computed(() => ({
  sales: '新增真实销售数据',
  weather: '天气数据',
  calendar: '校历数据',
}[kind.value]))

const summaryMetrics = computed(() => {
  if (!preview.value?.summary) return []
  const summary = preview.value.summary
  if (kind.value === 'sales') {
    return [
      ['原始行数', summary.original_rows ?? 0],
      ['有效行数', summary.valid_rows ?? 0],
      ['重复行数', summary.duplicate_rows ?? 0],
      ['异常行数', summary.abnormal_rows ?? 0],
      ['新菜数量', summary.new_dish_count ?? 0],
      ['未匹配菜品', summary.unmatched_dish_count ?? 0],
      ['缺失天气日期', summary.missing_weather_dates?.length ?? 0],
      ['缺失校历日期', summary.missing_calendar_dates?.length ?? 0],
    ]
  }
  return [
    ['原始行数', summary.original_rows ?? 0],
    ['有效行数', summary.valid_rows ?? 0],
    ['异常行数', summary.abnormal_rows ?? 0],
    ['新增日期', summary.new_dates?.length ?? 0],
    ['已存在日期', summary.existing_dates?.length ?? 0],
    ['冲突日期', summary.conflict_dates?.length ?? 0],
  ]
})
const previewColumns = computed(() =>
  Object.keys(preview.value?.rows?.[0] || {}).slice(0, 16),
)

function resetPreview() {
  preview.value = null
  sheetName.value = ''
  progress.value = 0
}

function clearSelectedUpload() {
  uploadRef.value?.clearFiles()
  uploadFiles.value = []
  selectedFile.value = null
}

function fileChanged(uploadFile) {
  selectedFile.value = uploadFile.raw
  resetPreview()
}

function fileRemoved(_uploadFile, currentFiles) {
  if (currentFiles.length) return
  selectedFile.value = null
  resetPreview()
}

function kindChanged() {
  clearSelectedUpload()
  resetPreview()
}

async function runPreview() {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择 CSV 或 XLSX 文件')
    return
  }
  if (kind.value === 'sales' && !datasetId.value) {
    ElMessage.warning('销售数据必须先选择逻辑食堂')
    return
  }
  previewing.value = true
  progress.value = 22
  try {
    const data = await previewUpload(
      kind.value,
      selectedFile.value,
      kind.value === 'sales' ? datasetId.value : null,
      sheetName.value || null,
    )
    progress.value = data.requires_sheet_selection ? 35 : 100
    preview.value = data
    if (data.requires_sheet_selection) {
      ElMessage.info('Excel 包含多个工作表，请选择后再次预览')
    } else {
      ElMessage.success('预览处理完成，尚未写入正式数据')
    }
  } catch (error) {
    ElMessage.error(errorMessage(error))
    progress.value = 0
  } finally {
    previewing.value = false
  }
}

async function commit() {
  if (!preview.value?.can_commit || !preview.value.preview_token) return
  committing.value = true
  try {
    const data = await commitPreview(kind.value, {
      preview_token: preview.value.preview_token,
      dataset_id: kind.value === 'sales' ? datasetId.value : null,
    })
    ElMessage.success(data.message)
    clearSelectedUpload()
    resetPreview()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    committing.value = false
  }
}

onMounted(async () => {
  try {
    Object.assign(options, await getOptions())
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
})
</script>

<template>
  <div>
    <div class="page-title">
      <div>
        <h1>数据更新</h1>
        <p>所有上传先预览、后确认；原始 sales.csv 保持只读，新增数据安全写入运行时文件。</p>
      </div>
      <el-tag type="warning" effect="plain">不会静默覆盖已有记录</el-tag>
    </div>

    <div class="update-layout">
      <el-card class="section-card upload-panel">
        <h2 class="section-heading"><el-icon><UploadFilled /></el-icon>上传与预览</h2>
        <el-form label-position="top">
          <el-form-item label="更新类型">
            <el-radio-group v-model="kind" @change="kindChanged">
              <el-radio-button value="sales">实际销售</el-radio-button>
              <el-radio-button value="weather">天气</el-radio-button>
              <el-radio-button value="calendar">校历</el-radio-button>
            </el-radio-group>
          </el-form-item>
          <el-form-item v-if="kind === 'sales'" label="逻辑食堂">
            <el-select v-model="datasetId" placeholder="请选择">
              <el-option
                v-for="item in options.datasets"
                :key="item.id"
                :label="item.label"
                :value="item.id"
              />
            </el-select>
          </el-form-item>
          <el-form-item :label="`${uploadTitle}文件`">
            <el-upload
              ref="uploadRef"
              v-model:file-list="uploadFiles"
              drag
              action="#"
              accept=".csv,.xlsx"
              :auto-upload="false"
              :limit="1"
              :on-change="fileChanged"
              :on-remove="fileRemoved"
            >
              <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
              <div class="el-upload__text">拖入或点击选择 <em>CSV / XLSX</em></div>
              <template #tip>
                <div class="el-upload__tip">最大30MB；不接受 XLS、XLSM、宏或公式。</div>
              </template>
            </el-upload>
          </el-form-item>
          <div v-if="preview?.requires_sheet_selection" class="sheet-select">
            <el-select v-model="sheetName" placeholder="选择 Excel 工作表">
              <el-option
                v-for="item in preview.sheet_names"
                :key="item"
                :label="item"
                :value="item"
              />
            </el-select>
          </div>
          <el-progress v-if="progress" :percentage="progress" :status="progress === 100 ? 'success' : ''" />
          <p class="progress-copy">
            预览阶段只写入短期暂存区，不会修改正式 CSV。
          </p>
          <el-button
            type="primary"
            :loading="previewing"
            :disabled="!selectedFile"
            style="width: 100%"
            @click="runPreview"
          >
            {{ preview?.requires_sheet_selection ? '按所选工作表重新预览' : '处理并预览' }}
          </el-button>
        </el-form>
      </el-card>

      <el-card class="section-card">
        <h2 class="section-heading"><el-icon><DocumentChecked /></el-icon>处理结果</h2>
        <template v-if="preview && !preview.requires_sheet_selection">
          <div class="metric-grid">
            <div v-for="[label, value] in summaryMetrics" :key="label" class="metric-box">
              <span>{{ label }}</span><strong>{{ value }}</strong>
            </div>
          </div>
          <el-descriptions v-if="kind === 'sales'" :column="2" border size="small">
            <el-descriptions-item label="日期范围">
              {{ preview.summary.date_range?.join(' ～ ') || '无有效日期' }}
            </el-descriptions-item>
            <el-descriptions-item label="缺失天气日期">
              {{ preview.summary.missing_weather_dates?.join('、') || '无' }}
            </el-descriptions-item>
            <el-descriptions-item label="缺失校历日期">
              {{ preview.summary.missing_calendar_dates?.join('、') || '无' }}
            </el-descriptions-item>
            <el-descriptions-item label="未匹配菜品">
              {{ preview.summary.unmatched_dishes?.join('、') || '无' }}
            </el-descriptions-item>
          </el-descriptions>
          <div v-if="preview.issues.length" class="issues">
            <el-alert
              v-for="(issue, index) in preview.issues"
              :key="`${issue.type}-${index}`"
              :title="issue.message"
              :description="issue.rows?.length ? `涉及源文件行：${issue.rows.join('、')}` : ''"
              type="warning"
              :closable="false"
              show-icon
              style="margin-bottom: 8px"
            />
          </div>
          <h3>原始/标准化预览（最多20行）</h3>
          <el-table :data="preview.rows" border stripe height="410">
            <el-table-column
              v-for="column in previewColumns"
              :key="column"
              :prop="column"
              :label="column"
              min-width="140"
              show-overflow-tooltip
            />
          </el-table>
          <div class="action-row" style="margin-top: 18px">
            <el-tag :type="preview.can_commit ? 'success' : 'danger'">
              {{ preview.can_commit ? '可以确认写入' : '存在阻止写入的问题' }}
            </el-tag>
            <el-popconfirm
              title="确认将预览中的新增有效记录写入本地数据？"
              confirm-button-text="确认写入"
              cancel-button-text="取消"
              @confirm="commit"
            >
              <template #reference>
                <el-button
                  type="success"
                  :loading="committing"
                  :disabled="!preview.can_commit"
                >
                  确认写入
                </el-button>
              </template>
            </el-popconfirm>
          </div>
        </template>
        <div v-else class="empty-panel">
          <el-icon size="42"><Files /></el-icon>
          <p>选择数据类型和文件，完成预览后才能确认写入。</p>
        </div>
      </el-card>
    </div>
  </div>
</template>
