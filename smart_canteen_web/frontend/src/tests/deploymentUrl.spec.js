import { deploymentUrl } from '../utils/deploymentUrl'

describe('download links under a deployment prefix', () => {
  it('keeps the report path and query under the public prefix', () => {
    expect(deploymentUrl('/api/recommendations/123/download?format=csv', '/menu_recommend/'))
      .toBe('/menu_recommend/api/recommendations/123/download?format=csv')
  })
  it('supports root deployments and absent draft download links', () => {
    expect(deploymentUrl('/api/report', '/')).toBe('/api/report')
    expect(deploymentUrl(null, '/menu_recommend/')).toBeUndefined()
  })
})
