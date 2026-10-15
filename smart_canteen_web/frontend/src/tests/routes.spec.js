import { describe, expect, it } from 'vitest'
import { routes } from '@/router'

describe('页面与 API 路由隔离', () => {
  it('仅声明两个 Vue 页面路由', () => {
    expect(routes.map((route) => route.path)).toEqual(['/', '/data-update'])
    expect(routes.every((route) => !route.path.startsWith('/api'))).toBe(true)
  })
})
