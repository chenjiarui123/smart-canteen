import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const viewPath = resolve(process.cwd(), 'src/views/DataUpdateView.vue')
const source = readFileSync(viewPath, 'utf8')

describe('数据更新文件选择状态', () => {
  it('上传组件文件列表由页面状态控制', () => {
    expect(source).toContain('const uploadFiles = ref([])')
    expect(source).toContain('v-model:file-list="uploadFiles"')
    expect(source).toContain('ref="uploadRef"')
  })

  it('写入成功后清空上传组件和预览状态', () => {
    expect(source).toContain('function clearSelectedUpload()')
    expect(source).toContain('uploadRef.value?.clearFiles()')
    expect(source).toContain('uploadFiles.value = []')
    expect(source).toMatch(/ElMessage\.success\(data\.message\)[\s\S]*clearSelectedUpload\(\)[\s\S]*resetPreview\(\)/)
  })

  it('写入失败分支不会清空待重试文件', () => {
    const catchBlock = source.match(/async function commit\(\)[\s\S]*?catch \(error\) \{([\s\S]*?)\} finally/)
    expect(catchBlock?.[1]).not.toContain('clearSelectedUpload()')
  })
})
