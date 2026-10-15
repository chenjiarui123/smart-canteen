import axios from 'axios'

const client = axios.create({
  baseURL: `${import.meta.env.BASE_URL}api`,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
})

export function errorMessage(error) {
  return error?.response?.data?.detail || error?.message || '请求失败，请稍后重试'
}

export const getOptions = () => client.get('/options').then(({ data }) => data)
export const getContext = (date) =>
  client.get(`/context/${encodeURIComponent(date)}`).then(({ data }) => data)
export const createRecommendation = (payload) =>
  client.post('/recommend', payload).then(({ data }) => data)
export const finalizeRecommendation = (payload) =>
  client.post('/recommend/finalize', payload).then(({ data }) => data)

export async function previewUpload(kind, file, datasetId, sheetName) {
  const endpoint = kind === 'sales' ? 'data' : kind
  const body = new FormData()
  body.append('file', file)
  if (datasetId) body.append('dataset_id', datasetId)
  if (sheetName) body.append('sheet_name', sheetName)
  const { data } = await client.post(`/${endpoint}/preview`, body, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
  return data
}

export const commitPreview = (kind, payload) =>
  client
    .post(`/${kind === 'sales' ? 'data' : kind}/commit`, payload)
    .then(({ data }) => data)

export default client
